//
// Created by omer1 on 02/07/2024.
//
#include "MapReduceFramework.h"
#include <pthread.h>
#include <atomic>
#include <iostream>
#include <bitset> // todo remove
#include <cstdint> // todo remove
#include "Barrier/Barrier.h"
#define SYS_MSG_ERR_PREFIX "system error: "
#define PTHREAD_CREATE_MSG_ERR "creating thread failed."
#define ALLOC_MSG_ERR "memory allocation failed."
#define PTHREAD_JOIN_MSG_ERR "pthred join failed."
// todo: free all allocations before exit(1)
struct JobContext;
void
init_job_context(const MapReduceClient& client, const InputVec& inputVec, OutputVec& outputVec,
                 int multiThreadLevel, JobContext*& jobContext);
void check_allocation_pointer(void* pointer);
void* threadMapWrapper(void* arg);
void* mapping(void* arg);
void* wrappingFunc(void* arg);
void set_counter(std::atomic<uint64_t>* atomic_counter, stage_t stage, uint64_t total_work);

struct ThreadContext
{
    IntermediateVec* intermediate_vec;
    int id;
    JobContext* job_context;
};

struct JobContext
{
    JobState* state;
    pthread_t* threads;
    ThreadContext* thread_contexts;
    int num_threads;
    const Barrier* barrier;
    pthread_mutex_t emit3;
    pthread_mutex_t reduce;
    const MapReduceClient* client;
    std::atomic<uint64_t>* atomic_counter;
    OutputVec* output_vec;
    const InputVec* input_vec;
    bool calledWait;
};

void emit2(K2* key, V2* value, void* context)
{
    std::cout<<"emit start" << std::endl;

    ThreadContext* tc = static_cast<ThreadContext*> (context);
    // todo: need to check if null?
    tc->intermediate_vec->emplace_back(key,value);
    for(int i = 0;tc->intermediate_vec->size();i++)
    {
        std::cout << "(" << (tc->intermediate_vec->at(i).first) << ", "
                 << (tc->intermediate_vec->at(i).second) << ") ";
    }
    std::cout << std::endl;
    std::cout<<"emit finished" << std::endl;

}

void emit3(K3* key, V3* value, void* context)
{
}

JobHandle startMapReduceJob(const MapReduceClient& client,
                            const InputVec& inputVec, OutputVec& outputVec,
                            int multiThreadLevel)
{
    JobContext* jobContext;
    init_job_context(client, inputVec, outputVec, multiThreadLevel, jobContext);

    for (int i = 0; i < multiThreadLevel; ++i)
    {
        jobContext->thread_contexts[i].intermediate_vec = new IntermediateVec();
        check_allocation_pointer(jobContext->thread_contexts[i].intermediate_vec);
        jobContext->thread_contexts[i].id = i;
        jobContext->thread_contexts[i].job_context = jobContext;
    }
    // starts the map stage
    set_counter(jobContext->atomic_counter, MAP_STAGE, inputVec.size());

    for (int i = 0; i < multiThreadLevel; ++i)
    {
         if (pthread_create(&jobContext->threads[i], nullptr, wrappingFunc, &jobContext->thread_contexts[i])
            != 0)
        {
            std::cout << SYS_MSG_ERR_PREFIX << PTHREAD_CREATE_MSG_ERR << std::endl;
            exit(1);
        }
    }


    return static_cast<JobHandle>(jobContext);
}

void waitForJob(JobHandle job)
{
    JobContext* jobContext = static_cast<JobContext*>(job);
    if (jobContext->calledWait)
    {
        return;
    }
    for (int i = 0; i < jobContext->num_threads; ++i)
    {
        if (pthread_join(jobContext->threads[i], nullptr) != 0)
        {
            std::cout << SYS_MSG_ERR_PREFIX << PTHREAD_JOIN_MSG_ERR << std::endl;
            exit(1);
        }
    }
    jobContext->calledWait = true;
}

void getJobState(JobHandle job, JobState* state)
{
    std::cout<<"job state started" << std::endl;
    JobContext* job_context = static_cast<JobContext*>(job);

    uint64_t cur_atomic = (job_context->atomic_counter->load());
    uint64_t cur_count = (cur_atomic & (static_cast<uint64_t>(0xffffffff) >> 1));
    uint64_t cur_stage = (cur_atomic >> 62);

    uint64_t total_work = (job_context->input_vec->size() >> 31 ) &  ((static_cast<uint64_t>(0xffffffff) >> 1));

    if(cur_stage == UNDEFINED_STAGE)
    {
        std::cout<<"job state middle" << std::endl;

        state->percentage = 0;
        state->stage = UNDEFINED_STAGE;
        return;
    }


    state->percentage = (static_cast<float>(cur_count) / static_cast<float>(total_work)) * 100;
    state->stage = static_cast<stage_t>(cur_stage);
    std::cout<<"job state finished" << std::endl;

}

void closeJobHandle(JobHandle job)
{
    JobContext* jobContext = static_cast<JobContext*>(job);
    if(jobContext == nullptr)
    {
        return;
    }
    waitForJob(job);

    for (int i = 0; i < jobContext->num_threads; ++i)
    {
        delete jobContext->thread_contexts[i].intermediate_vec;
    }

    delete[] jobContext->thread_contexts;
    delete[] jobContext->threads;
    delete jobContext->barrier;
    delete jobContext->atomic_counter;
    pthread_mutex_destroy(&jobContext->emit3);
    pthread_mutex_destroy(&jobContext->reduce);
    delete jobContext;
}

// HELPER functions
void* wrappingFunc(void* arg)
{

    mapping(arg);
    return nullptr;
}

void* mapping(void* arg)
{
    std::cout<<"map started " << std::endl;

    ThreadContext* tc = static_cast<ThreadContext*>(arg);
    const InputVec& inputVec = *(tc->job_context->input_vec);

    while (true)
    {
        uint64_t old_val = *(tc->job_context->atomic_counter);
        tc->job_context->atomic_counter++;
        if (old_val >= inputVec.size())
        {
            break;
        }

        tc->job_context->client->map(inputVec[old_val].first, inputVec[old_val].second, tc);
    }
    std::cout<<"map end " << std::endl;


    return nullptr;
}

void init_job_context(const MapReduceClient& client, const InputVec
                      & inputVec, OutputVec& outputVec, int multiThreadLevel, JobContext*& jobContext)
{
    jobContext = new JobContext;
    check_allocation_pointer(jobContext);

    jobContext->threads = new pthread_t[multiThreadLevel];
    check_allocation_pointer((void*)(jobContext->threads));

    jobContext->num_threads = multiThreadLevel;

    jobContext->state = new JobState;
    check_allocation_pointer(jobContext->state);
    jobContext->state->stage = UNDEFINED_STAGE;
    jobContext->state->percentage = 0;

    jobContext->client = &client;
    jobContext->input_vec = &inputVec;
    jobContext->output_vec = &outputVec;
    jobContext->calledWait = false;

    jobContext->atomic_counter = new std::atomic<uint64_t>(0);
    check_allocation_pointer((void*)(jobContext->atomic_counter));

    jobContext->barrier = new Barrier(multiThreadLevel);
    // todo: update mutex
    pthread_mutex_init(&jobContext->emit3, nullptr);
    pthread_mutex_init(&jobContext->reduce, nullptr);

    jobContext->thread_contexts = new ThreadContext[multiThreadLevel];
    check_allocation_pointer((void*)(jobContext->thread_contexts));
}

void check_allocation_pointer(void* pointer)
{
    if (pointer == nullptr)
    {
        std::cout << SYS_MSG_ERR_PREFIX << ALLOC_MSG_ERR << std::endl;
        exit(1);
    }
}

void set_counter(std::atomic<uint64_t>* atomic_counter, stage_t stage,
                 uint64_t total_work)
{
    // Zero the first 62 bits and preserve the last 2 bits
    uint64_t new_value =
        static_cast<uint64_t>(stage) << 62 & static_cast<uint64_t>(0x03) << 62;
    new_value  = new_value | ((static_cast<uint64_t>(total_work)<<31));
    atomic_counter->store(new_value);

}

