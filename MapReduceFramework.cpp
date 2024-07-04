//
// Created by omer1 on 02/07/2024.
//
#include "MapReduceFramework.h"
#include <pthread.h>
#include <atomic>
#include <iostream>
#include <bitset> // todo remove
#include <cstdint> // todo remove
#include<algorithm>
#include "Barrier/Barrier.h"
#define SYS_MSG_ERR_PREFIX "system error: "
#define PTHREAD_CREATE_MSG_ERR "creating thread failed."
#define ALLOC_MSG_ERR "memory allocation failed."
#define PTHREAD_JOIN_MSG_ERR "pthred join failed."
#define GET_COUNT &( static_cast<uint64_t>(0x7fffffff))
// todo: free all allocations before exit(1)
struct JobContext;
struct ThreadContext;
void

init_job_context(const MapReduceClient& client, const InputVec& inputVec, OutputVec& outputVec,
                 int multiThreadLevel, JobContext*& jobContext);
void check_allocation_pointer(void* pointer);
void* threadMapWrapper(void* arg);
void* mapping(void* arg);
void* wrappingFunc(void* arg);
void set_counter(std::atomic<uint64_t>* atomic_counter, stage_t stage, uint64_t total_work,
                 std::atomic<uint32_t>* started);
void sorting(ThreadContext* tc);

struct ThreadContext
{
    IntermediateVec* intermediate_vec;
    int id;
    JobContext* job_context;
};

struct JobContext
{
    pthread_t* threads;
    ThreadContext* thread_contexts; // to do change to vector ?
    int num_threads;
    const Barrier* barrier;
    pthread_mutex_t emit3;
    pthread_mutex_t reduce;
    const MapReduceClient* client;
    std::atomic<uint64_t> atomic_counter;
    std::atomic<uint32_t> started;
    OutputVec* output_vec;
    const InputVec* input_vec;
    bool calledWait;
};

void emit2(K2* key, V2* value, void* context)
{
    ThreadContext* tc = static_cast<ThreadContext*>(context);
    tc->intermediate_vec->emplace_back(key, value);
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
    set_counter(&jobContext->atomic_counter, MAP_STAGE, inputVec.size(), &jobContext->started);

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
    JobContext* job_context = static_cast<JobContext*>(job);

    uint64_t cur_atomic = (job_context->atomic_counter.load());
    uint64_t cur_count = (cur_atomic GET_COUNT);
    uint64_t cur_stage = (cur_atomic >> 62);

    uint64_t total_work = (cur_atomic >> 31) GET_COUNT;

    if (cur_stage == UNDEFINED_STAGE)
    {
        state->percentage = 0;
        state->stage = UNDEFINED_STAGE;
        return;
    }

    state->percentage = (static_cast<float>(cur_count) / static_cast<float>(total_work)) * 100;
    state->stage = static_cast<stage_t>(cur_stage);
}


void closeJobHandle(JobHandle job)
{
    JobContext* jobContext = static_cast<JobContext*>(job);
    if (jobContext == nullptr)
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
    pthread_mutex_destroy(&jobContext->emit3);
    pthread_mutex_destroy(&jobContext->reduce);
    delete jobContext;
}

// HELPER functions
void* wrappingFunc(void* arg)
{
    mapping(arg);
    // setting barrier
    return nullptr;
}

void* mapping(void* arg)
{
    ThreadContext* tc = static_cast<ThreadContext*>(arg);
    const InputVec& inputVec = *(tc->job_context->input_vec);

    while (true)
    {
        // need to get only the value of the counter
        uint64_t old_val = (tc->job_context->started)++;
        if (old_val >= inputVec.size())
        {
            break;
        }
        tc->job_context->client->map(inputVec[old_val].first, inputVec[old_val].second, tc);
        tc->job_context->atomic_counter.fetch_add(1);
        sorting(tc);
    }

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

    jobContext->client = &client;
    jobContext->input_vec = &inputVec;
    jobContext->output_vec = &outputVec;
    jobContext->calledWait = false;

    jobContext->atomic_counter = {0}; //TODO:CHANGED FROM POINTER TO VALUE
    jobContext->started = {0};


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
                 uint64_t total_work, std::atomic<uint32_t>* started)
{
    // Zero the first 62 bits and preserve the last 2 bits
    uint64_t new_value =
        static_cast<uint64_t>(stage) << 62 & static_cast<uint64_t>(0x03) << 62;
    new_value = new_value | ((static_cast<uint64_t>(total_work) << 31));
    atomic_counter->store(new_value);
    started->store(0);
}

void sorting(ThreadContext* tc)
{
    IntermediateVec* intermediate_vec = tc->intermediate_vec;
    std::sort(intermediate_vec->begin(),
              intermediate_vec->end(),
              [](const IntermediatePair& a, const IntermediatePair& b) { return *a.first < *b.first; });
    // locking
}
