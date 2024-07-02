//
// Created by omer1 on 02/07/2024.
//
#include "MapReduceFramework.h"
#include <pthread.h>
#include <atomic>
#include <iostream>
#include "Barrier/Barrier.h"
#define SYS_MSG_ERR_PREFIX "system error: "
#define PTHREAD_CREATE_MSG_ERR "creating thread failed."
#define ALLOC_MSG_ERR "memory allocation failed."
#define PTHREAD_JOIN_MSG_ERR "pthred join failed."
// todo: free all allocations before exit(1)
struct JobContext;
void
init_job_context (const MapReduceClient &client, const InputVec &inputVec, OutputVec &outputVec,
                  int multiThreadLevel, JobContext *&jobContext);
void check_allocation_pointer (void *pointer);
void *threadMapWrapper (void *arg);
void *mapping (void *arg);
void *wrappingFunc (void *arg);
void set_counter (std::atomic<uint64_t> *atomic_counter, stage_t stage);

struct ThreadContext
{
    IntermediateVec *intermediate_vec;
    int id;
    JobContext *job_context;
};

struct JobContext
{
    JobState *state;
    pthread_t *threads;
    ThreadContext *thread_contexts;
    int num_threads;
    const Barrier *barrier;
    pthread_mutex_t emit3;
    pthread_mutex_t reduce;
    const MapReduceClient *client;
    std::atomic<uint64_t> *atomic_counter;
    OutputVec *output_vec;
    const InputVec *input_vec;
    bool calledWait;
};

void emit2 (K2 *key, V2 *value, void *context)
{
}

void emit3 (K3 *key, V3 *value, void *context)
{
}

JobHandle startMapReduceJob (const MapReduceClient &client,
                             const InputVec &inputVec, OutputVec &outputVec,
                             int multiThreadLevel)
{
  JobContext *jobContext;
  init_job_context (client, inputVec, outputVec, multiThreadLevel, jobContext);

  for (int i = 0; i < multiThreadLevel; ++i)
  {
    jobContext->thread_contexts[i].intermediate_vec = new IntermediateVec ();
    check_allocation_pointer (jobContext->thread_contexts[i].intermediate_vec);
    jobContext->thread_contexts[i].id = i;
    jobContext->thread_contexts[i].job_context = jobContext;
  }

  for (int i = 0; i < multiThreadLevel; ++i)
  {
    if (pthread_create (&jobContext->threads[i], nullptr, wrappingFunc, &jobContext->thread_contexts[i])
        != 0)
    {
      std::cout << SYS_MSG_ERR_PREFIX << PTHREAD_CREATE_MSG_ERR << std::endl;
      exit (1);
    }
  }

  return static_cast<JobHandle>(jobContext);
}

void waitForJob (JobHandle job)
{
  JobContext *jobContext = static_cast<JobContext *>(job);
  if (jobContext->calledWait)
  {
    return;
  }
  for (int i = 0; i < jobContext->num_threads; ++i)
  {
    if (pthread_join (jobContext->threads[i], nullptr) != 0)
    {
      std::cout << SYS_MSG_ERR_PREFIX << PTHREAD_JOIN_MSG_ERR << std::endl;
      exit (1);
    }
  }
  jobContext->calledWait = true;
}

void getJobState (JobHandle job, JobState *state)
{
  JobContext *job_context = static_cast<JobContext *>(job);
  uint64_t cur_atomic = *(job_context->atomic_counter);
  uint64_t cur_count = (cur_atomic & 0xFFFFFFFFFFFFFFFC);
  uint64_t cur_stage = ((static_cast<uint64_t>(0x30) << 62) & cur_atomic);
  uint64_t total_work = job_context->input_vec->size () << 33 // TODO
  state->percentage = cur_count / job_context-> // TODO

}

void closeJobHandle (JobHandle job)
{

}

// HELPER functions
void *wrappingFunc (void *arg)
{
  mapping (arg);
  return nullptr;
}
void *mapping (void *arg)
{

  ThreadContext *tc = static_cast<ThreadContext *>(arg);
  const InputVec &inputVec = *(tc->job_context->input_vec);
  set_counter (tc->job_context->atomic_counter, MAP_STAGE);

  while (true)
  {
    uint64_t old_val = *(tc->job_context->atomic_counter);
    tc->job_context->atomic_counter++;
    if (old_val >= inputVec.size ())
    {
      break;
    }

    tc->job_context->client->map (inputVec[old_val].first, inputVec[old_val].second, tc);
  }

  return nullptr;
}

void init_job_context (const MapReduceClient &client, const InputVec
&inputVec, OutputVec &outputVec, int multiThreadLevel, JobContext *&jobContext)
{
  jobContext = new JobContext;
  check_allocation_pointer (jobContext);

  jobContext->threads = new pthread_t[multiThreadLevel];
  check_allocation_pointer ((void *) (jobContext->threads));

  jobContext->num_threads = multiThreadLevel;

  jobContext->state = new JobState;
  check_allocation_pointer (jobContext->state);
  jobContext->state->stage = UNDEFINED_STAGE;
  jobContext->state->percentage = 0;

  jobContext->client = &client;
  jobContext->input_vec = &inputVec;
  jobContext->output_vec = &outputVec;
  jobContext->calledWait = false;

  jobContext->atomic_counter = new std::atomic<uint64_t> (0);
  check_allocation_pointer ((void *) (jobContext->atomic_counter));

  jobContext->barrier = new Barrier (multiThreadLevel);
  // todo: update mutex
  pthread_mutex_init (&jobContext->emit3, nullptr);
  pthread_mutex_init (&jobContext->reduce, nullptr);

  jobContext->thread_contexts = new ThreadContext[multiThreadLevel];
  check_allocation_pointer ((void *) (jobContext->thread_contexts));
}

void check_allocation_pointer (void *pointer)
{
  if (pointer == nullptr)
  {
    std::cout << SYS_MSG_ERR_PREFIX << ALLOC_MSG_ERR << std::endl;
    exit (1);
  }
}

void set_counter (std::atomic<uint64_t> *atomic_counter, stage_t stage,
                  uint64_t total_work)
{
  // Zero the first 62 bits and preserve the last 2 bits
  uint64_t new_value =
      static_cast<uint64_t>(stage) << 62 & static_cast<uint64_t>(0x03) << 62;
// TODO save the total in 64 bit
  new_value = new_value | total_work << 32;
  atomic_counter->store (new_value);
}
