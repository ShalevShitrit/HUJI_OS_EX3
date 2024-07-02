//
// Created by omer1 on 02/07/2024.
//
#include "MapReduceFramework.h"




void emit2 (K2* key, V2* value, void* context);


void emit3 (K3* key, V3* value, void* context);

JobHandle startMapReduceJob(const MapReduceClient& client,
    const InputVec& inputVec, OutputVec& outputVec,
    int multiThreadLevel);

void waitForJob(JobHandle job);
void getJobState(JobHandle job, JobState* state);
void closeJobHandle(JobHandle job);


