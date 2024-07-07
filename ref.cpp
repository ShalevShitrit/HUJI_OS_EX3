//
// Created by omer1 on 06/07/2024.
//
/***** Includes *****/

#include <algorithm>
#include <iostream>
#include <atomic>
#include <pthread.h>
#include <cstdio>
#include "Barrier/Barrier.cpp"
#include "MapReduceFramework.h"

/***** Messages *****/

#define MSG_ERROR "system error: "
#define MSG_ALLOC_FAIL "Allocation failure."
#define MSG_MUTEX_INIT_FAIL "Pthread mutex init failure."
#define MSG_MUTEX_LOCK_FAIL "Pthread mutex lock failure."
#define MSG_MUTEX_UNLOCK_FAIL "Pthread mutex lock failure."
#define MSG_MUTEX_DESTROY_FAIL "Pthread mutex destroy failure."
#define MSG_PTHREAD_CREATION_FAIL "Pthread creation failure."
#define MSG_PTHREAD_JOIN_FAIL "Pthread join failure."

/***** Constants *****/
struct AtomicValues {
    uint64_t counter;
    uint64_t total;
    uint64_t stage;
};
#define FAIL 1
#define PERCENT_COEF 100
#define STAGE_LOC << 62
#define TOTAL_LOC << 31
#define GET_STAGE_BITS >> 62
#define GET_TOTAL_BITS >> 31 & (0x7fffffff)
#define GET_COUNT_BITS & (0x7fffffff)

/***** Typedefs *****/


typedef struct Job {
    pthread_t *workers{};
    Barrier *barrier{};
    pthread_mutex_t emit3_mutex{}, sort_mutex{};
    std::vector<IntermediateVec> intermediate_vecs;
    std::vector<IntermediateVec> shuffled_intermediate_vecs;
    JobState job_state{};
    std::atomic<uint64_t> state_atomic_counter{};
    std::atomic<uint32_t> to_process_counter{};
    int multiThreadLevel{};
    const MapReduceClient *client{};
    const InputVec *inputVec{};
    OutputVec *outputVec{};
    bool called_join{};
} Job;

typedef struct ThreadContext {
    int threadID{};
    IntermediateVec intermediate_vec;
    Job *job{};
} ThreadContext;

typedef struct JobContext {
    ThreadContext *contexts;
    Job *job;
} JobContext;

/***** Global Variables *****/




/***** General Helper Functions *****/

/*
 * Returns the data within the 64bit input number.
 */
AtomicValues get_atomic_values (uint64_t number)
{
  auto counter = number GET_COUNT_BITS;
  auto total = number GET_TOTAL_BITS;
  auto stage = number GET_STAGE_BITS;
  return {counter, total, stage};
}

/*
 * Returns true if first equals second and false otherwise
 */
bool is_equal (const K2 *first, const K2 *second)
{
  return !(*first < *second || *second < *first);
}

/*
 * Returns the max key that is in intermediate_vecs.
 * If intermediate_vecs is empty the function returns nullptr.
 */
K2 *max_key (Job *job)
{
  // check if intermediate_vecs is empty
  bool is_empty = true;
  for (const auto &vec: job->intermediate_vecs)
    is_empty = is_empty && vec.empty ();
  if (is_empty) return nullptr;

  // find the max key
  // get the first max key
  K2 *max_value;
  for (const auto &vec: job->intermediate_vecs)
    {
      if (!vec.empty ())
        {
          max_value = vec.back ().first;
          break;
        }
    }
  // look for a greater key
  for (const auto &vec: job->intermediate_vecs)
    if (!vec.empty () && *max_value < *(vec.back ().first))
      max_value = vec.back ().first;
  return max_value;
}

/*
 * Counts the number of elements in the vector of vectors
 */
size_t count_elements (std::vector<IntermediateVec> &vec)
{
  size_t counter = 0;
  for (auto &v: vec)
    {
      counter += v.size ();
    }
  return counter;
}

/*
 * Sets the atomic counter to zero and sets its stage and total.
 */
void reset_counter (Job *job, stage_t stage, uint64_t total)
{
  job->state_atomic_counter = ((uint64_t) stage STAGE_LOC) |
                              (total TOTAL_LOC);
  job->to_process_counter = 0;
}

/***** Phases Functions *****/

/*
 * Each thread will sort its intermediate vector according to the keys within
 */
void sort_phase (ThreadContext *thread_context)
{
  auto *curr_vec = &thread_context->intermediate_vec;
  if (curr_vec->empty ()) return;

  std::sort (curr_vec->begin (), curr_vec->end (),
             [] (const IntermediatePair &a, const IntermediatePair &b)
             { return *a.first < *b.first; });

  // pushing to a shared vector
  if(pthread_mutex_lock (&thread_context->job->sort_mutex))
    {
      std::cout << MSG_ERROR << MSG_MUTEX_LOCK_FAIL << std::endl;
      exit (FAIL);
    }
  thread_context->job->intermediate_vecs.push_back (*curr_vec);
  if(pthread_mutex_unlock (&thread_context->job->sort_mutex))
    {
      std::cout << MSG_ERROR << MSG_MUTEX_UNLOCK_FAIL << std::endl;
      exit (FAIL);
    }
}

/*
 * In this phase each thread reads pairs of (k1, v1) from the input vector and
 * calls the map function on each of them. The map function in turn will produce
 * (k2, v2) and will call the emit2 function to update the framework databases.
 */
void map_phase (ThreadContext *thread_context)
{
  auto *curr_job = thread_context->job;
  while ((curr_job->state_atomic_counter.load () GET_COUNT_BITS)
         < (curr_job->state_atomic_counter GET_TOTAL_BITS))
    {
      auto old_value = curr_job->to_process_counter++;
      if (old_value >= curr_job->inputVec->size ()) break;
      InputPair pair = curr_job->inputVec->at (old_value);
      // call map
      curr_job->client->map (pair.first, pair.second, thread_context);
      curr_job->state_atomic_counter++;
    }
  // call sort
  sort_phase (thread_context);
}



/* Thread number 0 will combine the vectors into a single data structure.
 * The function uses an atomic counter for counting the number of vectors in it.
 */
void shuffle_phase (ThreadContext *thread_context)
{
  auto *curr_job = thread_context->job;
  K2 *max_key_element = max_key (thread_context->job);

  // for every max key
  while (max_key_element)
    {
      IntermediateVec pairs_vec;
      // go over the intermediate vectors
      for (auto &vec: thread_context->job->intermediate_vecs)
        {
          // go over the vector and collect the max keys that are equal to max_key_element
          while (!vec.empty () &&
                 is_equal (vec.back ().first, max_key_element))
            {
              pairs_vec.push_back (vec.back ());
              vec.pop_back ();
              curr_job->state_atomic_counter++;
            }
        }
      thread_context->job->shuffled_intermediate_vecs.push_back (pairs_vec);
      max_key_element = max_key (thread_context->job);
    }
}

/*
 * the function pops a vector from the back of the queue and run reduce on it.
 */
void reduce_phase (ThreadContext *thread_context)
{
  auto *curr_job = thread_context->job;
  while ((curr_job->state_atomic_counter.load () GET_COUNT_BITS)
         < (curr_job->state_atomic_counter.load () GET_TOTAL_BITS))
    {
      auto old_value = curr_job->to_process_counter++;
      if (old_value >= thread_context->job->shuffled_intermediate_vecs.size ()) break;
      IntermediateVec shuffled_vec =
          thread_context->job->shuffled_intermediate_vecs.at (old_value);
      // call reduce
      thread_context->job->client->reduce (&shuffled_vec, thread_context);
      curr_job->state_atomic_counter++;
    }

}

/***** Thread And Job Related Functions *****/

/*
 * Worker is the thread function.
 * The workers are applying the phases on the data and filling the output
 * vector of the job.
 */
void *worker (void *arg)
{
  auto *thread_context = (ThreadContext *) arg;

  map_phase (thread_context);

  /* ----- Barrier between phases ----- */
  thread_context->job->barrier->barrier ();

  // only thread 0 can shuffle
  if (thread_context->threadID == 0)
    {
      reset_counter (thread_context->job, SHUFFLE_STAGE,
                     count_elements (thread_context->job->intermediate_vecs));

      shuffle_phase (thread_context); // call shuffle

      reset_counter (thread_context->job, REDUCE_STAGE,
                     thread_context->job->shuffled_intermediate_vecs.size ());
    }

  /* ----- Barrier between phases ----- */
  thread_context->job->barrier->barrier ();

  reduce_phase (thread_context);

  return (void *) thread_context;
}

/*
 * Initializer of the job.
 * Allocates all the data and sets the threads.
 */
JobContext *job_init (const MapReduceClient *client,
                      const InputVec *inputVec,
                      OutputVec *outputVec,
                      int multiThreadLevel)
{
  // allocate the job context
  auto *job_context = new (std::nothrow) JobContext;
  if (job_context == nullptr)
    {
      std::cout << MSG_ERROR << MSG_ALLOC_FAIL << std::endl;
      exit (FAIL);
    }

  /* set job data */
  job_context->job = new (std::nothrow) Job;
  if (job_context->job == nullptr)
    {
      std::cout << MSG_ERROR << MSG_ALLOC_FAIL << std::endl;
      exit (FAIL);
    }

  job_context->job->workers = new (std::nothrow) pthread_t[multiThreadLevel];
  if (job_context->job->workers == nullptr)
    {
      std::cout << MSG_ERROR << MSG_ALLOC_FAIL << std::endl;
      exit (FAIL);
    }
  job_context->job->barrier = new (std::nothrow) Barrier (multiThreadLevel);
  if (job_context->job->barrier == nullptr)
    {
      std::cout << MSG_ERROR << MSG_ALLOC_FAIL << std::endl;
      exit (FAIL);
    }

  job_context->job->job_state = {UNDEFINED_STAGE, 0};
  job_context->job->called_join = false;

  // these are the input data:
  job_context->job->multiThreadLevel = multiThreadLevel;
  job_context->job->client = client;
  job_context->job->inputVec = inputVec;
  job_context->job->outputVec = outputVec;

  // set the mutexes
  if (pthread_mutex_init(&job_context->job->emit3_mutex, nullptr)){
      std::cout << MSG_ERROR << MSG_MUTEX_INIT_FAIL << std::endl;
      exit (FAIL);
  }
  if (pthread_mutex_init(&job_context->job->sort_mutex, nullptr)){
      std::cout << MSG_ERROR << MSG_MUTEX_INIT_FAIL << std::endl;
      exit (FAIL);
    }

  /* set threads data */
  job_context->contexts = new (std::nothrow) ThreadContext[multiThreadLevel];
  if (job_context->contexts != nullptr)
    {
      std::cout << MSG_ERROR << MSG_ALLOC_FAIL << std::endl;
      exit (FAIL);
    }

  for (int i = 0; i < multiThreadLevel; ++i)
    {
      job_context->contexts[i].threadID = i;
      job_context->contexts[i].job = job_context->job;
    }

  return job_context;
}

/***** External Interface *****/


/*
 * The function receives as input intermediary element (K2, V2) and context.
 * The function saves the intermediary element in the context data structures and
 * updates the number of intermediary elements using atomic counter.
 */
void emit2 (K2 *key, V2 *value, void *context)
{
  auto *thread_context = (ThreadContext *) context;
  thread_context->intermediate_vec.push_back ({key, value});
}

/*
 * The function receives as input intermediary element (K3, V3) and context.
 * The function saves the output element in the context data structures
 * (output vector) and updates the number of output elements using atomic counter.
 */
void emit3 (K3 *key, V3 *value, void *context)
{
  auto *thread_context = (ThreadContext *) context;
  if(pthread_mutex_lock (&thread_context->job->emit3_mutex))
    {
      std::cout << MSG_ERROR << MSG_MUTEX_LOCK_FAIL << std::endl;
      exit (FAIL);
    }
  thread_context->job->outputVec->push_back ({key, value});

  if(pthread_mutex_unlock (&thread_context->job->emit3_mutex))
    {
      std::cout << MSG_ERROR << MSG_MUTEX_UNLOCK_FAIL << std::endl;
      exit (FAIL);
    }
}

/**
 * Starts the multi threaded map-reduce job.
 * @param client - The map-reduce api.
 * @param inputVec - The input vector of the job.
 * @param outputVec - The vector to place there the output.
 * @param multiThreadLevel - number of threads.
 * @return job_context - The context of the running job.
 */
JobHandle startMapReduceJob (const MapReduceClient &client,
                             const InputVec &inputVec, OutputVec &outputVec,
                             int multiThreadLevel)
{
  // job init
  auto *job_context = job_init (&client,
                                &inputVec,
                                &outputVec,
                                multiThreadLevel);

  // switch the counter to map phase
  reset_counter (job_context->job, MAP_STAGE, job_context->job->inputVec->size ());

  // let the workers do their magic
  for (int i = 0; i < multiThreadLevel; ++i)
    {
      if(pthread_create (job_context->job->workers + i,
                      nullptr,
                      worker,
                      job_context->contexts + i)){
          std::cout << MSG_ERROR << MSG_PTHREAD_CREATION_FAIL << std::endl;
          exit (FAIL);
      }
    }

  return (JobHandle) job_context;
}

/**
 * This function lets the user wait until all the threads finish their job.
 * @param job - The job to wait
 */
void waitForJob (JobHandle job)
{
  auto *curr_job = ((JobContext *) job)->job;
  if (curr_job->called_join) return;
  for (int i = 0; i < curr_job->multiThreadLevel; ++i)
    {
      if (pthread_join (curr_job->workers[i], nullptr))
        {
          std::cout << MSG_ERROR << MSG_PTHREAD_JOIN_FAIL << std::endl;
          exit (FAIL);
        }
    }
  curr_job->called_join = true;
}

/**
 * This function lets the user to get the current state of the given job.
 * @param job - The job to get its state.
 * @param state - The place for the output state.
 */
void getJobState (JobHandle job, JobState *state)
{
  auto *curr_job = ((JobContext *) job)->job;
  auto values = get_atomic_values (curr_job->state_atomic_counter.load ());

  curr_job->job_state.stage = static_cast<stage_t>(values.stage);
  curr_job->job_state.percentage = PERCENT_COEF * (float) values.counter / (float) values.total;

  if (curr_job->job_state.stage == UNDEFINED_STAGE)
    curr_job->job_state.percentage = 0;

  if (curr_job->job_state.percentage > 100)
    curr_job->job_state.percentage = 100;

  *state = curr_job->job_state;
}

/**
 * Waits until the job is done and then deallocates all the job`s dynamically allocated data.
 * @param job - The job to close.
 */
void closeJobHandle (JobHandle job)
{
  waitForJob (job);
  auto *curr_job = (JobContext *) job;

  if(pthread_mutex_destroy (&curr_job->job->emit3_mutex))
    {
      std::cout << MSG_ERROR << MSG_MUTEX_DESTROY_FAIL << std::endl;
      exit (FAIL);
    }
  if(pthread_mutex_destroy (&curr_job->job->sort_mutex))
    {
      std::cout << MSG_ERROR << MSG_MUTEX_DESTROY_FAIL << std::endl;
      exit (FAIL);
    }

  delete[] curr_job->contexts;

  delete[] curr_job->job->workers;
  delete curr_job->job->barrier;
  delete curr_job->job;

  delete curr_job;
}