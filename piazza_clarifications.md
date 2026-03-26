Assignment 2 released.
Updated 2 days ago by Riti Verma
Added to reading list
Assignment 2 has been released on moodle. 

Any updates on the problem statement will be reflected on this thread. For all your queries, kindly use this thread only. 

Deadline is March 24, EOD. You are strongly advised to start early to avoid last minute clarifications.

Deadline has been extended to March 27, EOD.

------------------------------------------------------------------- Submission instructions ---------------------------------------------------------------

Submit a zip file of a folder named a2.

You can have any number of source files but must include a makefile. In addition to source files, there should also be a report.pdf.

The following commands will be run on your submission-

unzip a2
cd a2
make ( should produce an executable named a2 )
./a2 input.txt knn ( should produce the knn.txt file in the same directory )
./a2 input.txt approx_knn ( should produce the approx_knn.txt file in the same directory )
./a2 input.txt kmeans ( should produce the kmeans.txt file in the same directory )
Any submissions not following the above instructions WILL NOT be graded.

You may refer to the following resources for HPC usage-

https://supercomputing.iitd.ac.in/?access
https://github.com/kanha95/HPC-IIT-Delhi
This is a simple input.txt file generator (as requested by some students) which you can use to test your code.

dataset_generator.py

Vary the parameters (n, k, T) according to your need.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

You need to compute the local histogram over the neighborhood of a point. This neighborhood we define in three ways-

knn - the top k points near the center point. these k points do not include the point itself. now you need to compute the local histogram over this center point and its k nearest neighbours which means (k+1) points are considered for the histogram.
approximate knn - done in the same way
kmeans cluster - here you define k clusters. each cluster includes the point itself. compute a local histogram over each cluster (including the center point).
Update - In local intensity remapping, in case of KNN algorithms, use m = k+1.

We have decided to keep the mae_loss threshold to 3 for correctness. All submissions within this threshold will be considered for optimization. For the correctness marks, the submissions which are within the threshold and have lower loss will get more marks, so within a threshold, correctness marks will be relative to the submissions we receive.

Note that your code should yield lower mae loss than 3 for every input. We might consider increasing the threshold later if very less submissions satisfy the condition for a given input so you need not worry about that. 

You can use the below script to check for mae_loss-

mae_loss.py

PDP_A2.pdf

1
261 views
143 Followup Discussions

Resolved
Anonymous Gear  2 weeks ago
Is the assignment to be done in pairs or individually ?
0
Riti Verma  2 weeks ago
Individually
1

Resolved
Anonymous Mouse  2 weeks ago
Can the exact commands which will be run be told, file name formats and compilation expectations
What cuda version must we write for ?
2
Anonymous Mouse  2 weeks ago
Also can an example input-output pair/test generator be provided for correctness verification

Can we presume to match scikit-learn's knn to be ground truths?

1
Riti Verma  2 weeks ago
Why do you need scikit-learn's knn as ground truths? for exact KNN, you simply need to find the top k closest points according to Euclidean distance.
0

Resolved
Anonymous Scale  2 weeks ago
Will we get access to HPC for this assignment? Can you please share details on how to access?

4
Anonymous Scale  2 weeks ago
Please reply. Don’t have CUDA on mac

0
Kashish Goel  1 week ago
HPC access has already been provided. Please ssh using kerberos@hpc.iitd.ac.in and use kerberos pswd as the pswd.
Rest of the instructions can be seen from here - https://supercomputing.iitd.ac.in/?access
0
Anonymous Comp  1 week ago
Can you guide how can we compile and run .cu files, after ssh to hpc.iitd.ac.in I cannot find the nvcc compiler,also can we run directly on login nodes?
1
Riti Verma  1 week ago
You can start a job on hpc using this command-

qsub -I -P col7880.<kerberos>.course -N test -l select=1:ncpus=1:ngpus=1
Once the job starts, you will be shifted from login node to Haswell (khas) node.

Run the following to load cuda modules-

module purge
module load compiler/gcc/9.1.0
module load suite/nvidia-hpc-sdk/20.7/cuda11.0
This will load the necessary modules. module purge ensures that no extra modules are loaded.

You can compile and run your program using - 

nvcc test.cu -arch=sm_35 -Xcompiler -fopenmp -o test
./test
Use "exit" command to exit the job.

It is always a good practice to use "walltime" tag on the job to ensure a job doesn't keep running indefinitely (you might run out of hpc credits in this case).

NOTE - The project name "col7880" is common for all of you (including col380 and col730 students).

5
Anonymous Beaker 3  1 day ago
I understand that, in the serial version of code, against which you will test the performance of our code, you will not use the -O3 optimisation flag. However, in our Makefile, should we add this flag to optimise our parallel code? Using it does make quite a difference in execution time.

Here, you have given an example of how to compile on the HPC as follows:

nvcc test.cu -arch=sm_35 -Xcompiler -fopenmp -o test
Is this just a demo, or is this the standard we must follow when compiling our parallel version on the HPC? For e.g our compilation command must look like:

nvcc a2.cu -arch=sm_35 -Xcompiler -fopenmp -o a2
Also, hopefully, nvcc warnings related to sm_35 will be ignored while testing.

0
Riti Verma  1 day ago
You can use the O3 flags - add it in your report under optimizations.

"Is this just a demo" - It's just a demo. You are free to use any command in your makefile.

"nvcc warnings related to sm_35 will be ignored while testing" - Yes

1
Anonymous Calc 2  11 hours ago
Are we allowed to use sm_50 or sm_80 ? or is sm_35 strict?
0
Riti Verma  8 hours ago
sm_35
0

Resolved
Saurabh Singh  2 weeks ago
When can we expect the test file?
1
Saurabh Singh  1 week ago
I had discussed the issue with Sir and he has asked me to contact the TAs on email for sharing of the dataset generator code. Can someone share the email ID being used by the TAs.
0
Riti Verma  1 week ago
We have released the script.
0

Resolved
Anonymous Beaker  2 weeks ago
can we know the constraints on n,k,t ?
1
Riti Verma  2 weeks ago
n 

 k 
 

T 

3
Anonymous Beaker  1 week ago
Also can we assume the coordinates will be between -104 to 104  


2
Riti Verma  1 week ago
Yes.
0

Resolved
Abhinav Rajesh Shripad  2 weeks ago
If the starter code is not provided at the time the assignment is released, the deadline should be extended by the exact number of days by which the starter code was delayed.
6
Riti Verma  2 weeks ago
There is no starter code.
0
Anonymous Calc  2 weeks ago
How are we expected to know Input output format
0
Riti Verma  2 weeks ago
It is clearly mentioned in the pdf.
0

Resolved
Anonymous Poet  2 weeks ago
will you run this code on ubuntu or windows? I want to use <sys/mman.h> for reading file but it is a POSIX/Unix standard header and is not natively available on Windows.
0
Riti Verma  1 week ago
hpc
0

Resolved
Vanshika  2 weeks ago
Please provide details for accessing HPC. Or could you please tell some alternative for running my code on MAC?
3
Kashish Goel  1 week ago
refer to https://piazza.com/class/mk2exak3kau4fk/post/26_f3
0

Resolved
Anonymous Atom  2 weeks ago
IIn this definition will the point itself be included in this set?
0
Riti Verma  1 week ago
No
1
Anonymous Comp 4  6 hours ago
mam if the point itself will not be included then it can happen that the new intensity is coming to be negative, what do we have to do in that case
0
Riti Verma  6 hours ago
refer to https://piazza.com/class/mk2exak3kau4fk/post/26_f20
0

Resolved
Abhinav Rajesh Shripad  2 weeks ago
Sample Input/Output will be provided like Assignment 1 ?
0
Riti Verma  1 week ago
Released the input generator file. You can produce your own output files.
0

Resolved
Anonymous Helix  1 week ago
Just to confirm, since I couldn’t find it in the assignment PDF. Are we allowed to use C++ or only limited to C?

0
Riti Verma  1 week ago
You are free to use anything as long as it compiles with the following modules-

module load compiler/gcc/9.1.0
module load suite/nvidia-hpc-sdk/20.7/cuda11.0
Just make the suitable changes in your makefile.

1

Resolved
Anonymous Gear 2  1 week ago
As per instructions, a single executable will generate the three output files (for each approach), always (i.e. we do not specify any flag indicating which approach to use to generate a single output.txt file). So will the performance comparison with sequential version be done of all three approaches together (i.e. total time taken by all three), or their individual times will be compared?
1
Riti Verma  1 week ago
Individual times will be compared. Apologies for the confusion. I will share a slight modification to the run commands shortly.
3
Anonymous Gear 2  14 hours ago
Will the time taken by our code for taking input and writing output also considered, or only kernel time will be considered?

0

Resolved
Anonymous Helix  1 week ago
Since the assignment also expects OpenMP. Can we get an estimate of the number of cores allowed for a single run? This will help me resolve some crucial design decisions.

1
Riti Verma  1 week ago
It is not fixed. We will test on different number of cores.
0

Resolved
Anonymous Helix  1 week ago
For approx KNN, I understand that the threshold will be decided based on the submissions. Could the TAs give a rough estimate of the order of MAE that will be allowed/acceptable? This will help in design decisions (the trade-off between speedup and accuracy)

1
Riti Verma  1 week ago
Can't share beforehand. It depends on the submissions.
0
Anonymous Helix  1 week ago
I understand that it may be hard to set a threshold arbitrarily. I would ask you to kindly reconsider. This is because the Approximate KNN algorithm’s performance will only be graded if it is considered correct.

This effectively reduces the problem to just a game of blind blackjack. Where we have to choose an arbitrary MAE to build our solution around.

There are two ways:

As requested, let’s agree on the ORDER of the MAE error, the exact cut-offs can still be decided after submissions are in.
Grade the correctness and performance on a curve. Where even “incorrect” submissions are allowed to run with a penalty function parameterized by the MAE.
6
Riti Verma  2 days ago
We have decided to keep the mae_loss threshold to 3 for correctness. All submissions within this threshold will be considered for optimization. For the correctness marks, the submissions which are within the threshold and have lower loss will get more marks, so within a threshold, correctness marks will be relative to the submissions we receive.

Note that your code should yield lower mae loss than 3 for every input. We might consider increasing the threshold later if very less submissions satisfy the condition for a given input so you need not worry about that. 

You can use the below script to check for mae_loss-

mae_loss.py

1

Resolved
Anonymous Mouse 2  1 week ago
Will submissions be run on Haswell architecture and Tesla K40 gpu ? How many cpu's and gpu's we have been allocated in this project ?
2
Riti Verma  1 week ago
Yes. The number of cpus and gpus will be varied.
1

Resolved
Anonymous Scale 2  1 week ago
In case of tie between two points distance which point to choose first as this can cause issue when you have already got your k-1 neighbour points and then there is tie for the k-th place?
0
Riti Verma  1 week ago
Choose the lexically smaller point.

For A(x1, x2, x3) and B(y1, y2, y3),

choose A if

;
or 
 and 
 ;
or 
 and 
 and 
1
Anonymous Mouse 3  4 days ago
Will the input points be lexically sorted?

0
Riti Verma  3 days ago
no
0

Resolved
Anonymous Scale 2  1 week ago
Will the coordinates be integer valued or can they also be decimals?

 

0
Riti Verma  1 week ago
Integers
0

Resolved
Anonymous Beaker 2  1 week ago
on running:
$ ssh cs1230723@hpc.iitd.ac.in
cs1230723@hpc.iitd.ac.in's password: 

I am getting:
Connection closed by 10.113.30.13 port 22


do we have to take access from somewhere or is it given already? if so whats the issue here?

1
Riti Verma  1 week ago
Have you followed the steps in https://piazza.com/class/mk2exak3kau4fk/post/31?
0
Anonymous Beaker 2  1 week ago
oh thanks. its saying sir has to approve it. thats it right? after i get approval mail i am good to go right?
0
Riti Verma  1 week ago
yes
0

Unresolved
Anonymous Gear 2  1 week ago
In the kmeans algorithm can you please clarify like the centroids that we get should need to make them integers or we should treat them as floats or double please clarify on that also in case of tie what to do with points like which cluseter should we assign that to?
2
Riti Verma  6 days ago
Use integer division for computing the centroids.

Assign to the cluster with lexically smaller centroid.

0
Anonymous Beaker  6 days ago
I think it would be better if you can give us some implementations to check for correctness for kmeans especially, since the way we divide would change the value we get in case division has to be integer .

So by your implementation we could confirm that the values are atleast same. Also since he deadline is extended to 27th , you may do it later too.

Also one thing I feel is that even not-very-much optimised kmeans won't take much long in case n is limited to 1e5 . When  using n as 1e6 the difference was much better visible b/w an optimized and unoptimized code 

0
Riti Verma  6 days ago
You don't have to implement division separately. 

You just have to initialize the centroid coordinates as integers. 

int centroid = sum of coordinates(integer) / count(integer)

1
Anonymous Helix 3  16 hours ago
what about intensity remapping? should it be integer division too?
1
Riti Verma  8 hours ago
No. you need to cast them into double/float and then take floor of the same.

Do something like this - 

double temp = (((double)(C[Ii]-Cmin))/ (double)(m-Cmin))*255.0;
new_intensity[i] = (int)floor(temp);
0
Anonymous Scale 4  8 hours ago
But isn't that equivalent to doing ((C[Ii]-Cmin)*255) / (m-Cmin) where everything is an integer? Casting it explicitly is redundant here.
2
Anonymous Helix 3  8 hours ago
and also may cause floating point precision errors due to division, doing integer division gives more precise results

3
Anonymous Gear 4  6 hours ago
Mam please clarify on the above.
Because  ((C[Ii]-Cmin)*255) / (m-Cmin) is better (gives better precision) rather than solving floating arithimetic and using floor
2

Resolved
Anonymous Gear 2  1 week ago
the Nk(i) that we get in case of kmeans is the point i also considered there?

2
Anonymous Poet 2  7 days ago
Yes sir could you please clarify this point since this will change the behaviour of Exact KNN output.
1
Riti Verma  6 days ago
You need to compute the local histogram over the neighborhood of a point. This neighborhood we define in three ways-

knn - the top k points near the center point. these k points do not include the point itself. now you need to compute the local histogram over this center point and its k nearest neighbours which means (k+1) points are considered for the histogram.
approximate knn - done in the same way
kmeans cluster - here you define k clusters. each cluster includes the point itself. compute a local histogram over each cluster (including the center point).
I hope this clears things.

0
Anonymous Gear 2  6 days ago
Can you please also clarify, if k+1 points are considered for histogram, should we keep m in the I’ formula same as k (as in the instructions for assignment) or k+1 (as per this clarification)? Thanks.

0
Riti Verma  6 days ago
It should be k+1. Thanks for pointing it out.
0
Anonymous Comp 2  5 days ago
But it was specified differently in the pdf the Only Ni() only neighbourhood points are considered for cdf and intensity calculation. So are you saying to consider i into Ni() and cdf calculation.

0
Anonymous Gear 2  5 days ago
Can we assume n >= k+1 in KNN, and n >= k in K-Means?

0
Riti Verma  5 days ago
"But it was specified differently in the pdf the Only Ni() only neighbourhood points are considered for cdf and intensity calculation. So are you saying to consider i into Ni() and cdf calculation." -> Yes. Please consider the point as well as its k neighbourhood points while computing the histogram.

"Can we assume n >= k+1 in KNN, and n >= k in K-Means?" - Yes

0

Resolved
Nayan Rabadiya  6 days ago

in case of I' < 0  in above formula should we make it  I' = 0 to keep it in (0,255)?

0
Riti Verma  6 days ago
Yes.

Clip the I' to range between 0 to 255 (both inclusive).

2

Resolved
Anonymous Poet  6 days ago
I don't understand approximate KNN thing. We can just do anything random to minimize cuda use and make our code faster. If you are going to compare error then different algorithm may give different error compared to actual knn depending on dataset.
0
Riti Verma  6 days ago
If you do anything random to minimize cuda use, you will likely be increasing the approximation and end up with higher error. 

0

Resolved
Anonymous Calc 2  6 days ago
When the speedup is compared with sequential version, will the sequential version be compiled with -O3 flag or without it? Because it will lead to huge time difference in case of knn (dependent on implementation though).
0
Riti Verma  5 days ago
Without
0

Resolved
Anonymous Atom 2  6 days ago
Deadline postponed to 27th right?

It's not updated yet on new Moodle...

1
Riti Verma  5 days ago
Yes
2

Resolved
Anonymous Helix 2  4 days ago
- In k-means clustering, what initial points should we choose ? (as final answer may depend heavily on it)
- Also, it is given that for k-means clustering, T is number of iterations. But number of restarts is not mentioned(choosing different initial points) is not mentioned. So are we supposed to have a single restart ?
0
Riti Verma  4 days ago
It is mentioned in the pdf. You have to initialise using the first k points in the input.txt file. There are no restarts.
0

Resolved
Anonymous Gear 3  4 days ago
Can we assume ? Are all points are unique?

0
Riti Verma  3 days ago
yes
1

Unresolved
Nayan Rabadiya  2 days ago
Will GPU persistence mode be enabled during the test runs?
0

Unresolved
Anonymous Gear 2  2 days ago
Is anyone else facing the issue regarding insufficient memory on HPC at random times? How can it be resolved? Thanks.

0

Resolved
Anonymous Scale 2  2 days ago
Will the test cases follow same random distribution of points, as in the dataset_generator.py? Or will the test cases for final evaluaiton use different distributions?
0
Riti Verma  1 day ago
We cannot disclose that.
0

Resolved
Anonymous Gear 3  1 day ago
Hi,
I have a doubt regarding how ties are resolved in KNN.
Suppose k=2, and the query point is 0 0 0. Consider three training points: 0 0 1, 0 1 0, and 1 0 0. All of them are at the same distance from the query point.
In such a case, how do we decide which k points to select when multiple points have equal distances? Also, if this leads to a tie in class labels, what is the standard approach to break the tie?
0
Anonymous Scale 3  1 day ago
https://piazza.com/class/mk2exak3kau4fk/post/26_f16
1
Endorsed by Instructor (Riti Verma)

Resolved
Anonymous Beaker  1 day ago
Will our approx. KNN speedup be tested based on our KNN? or it's over the sequential version
Because due to n being only up to 1e5, a major speedup is not observed.
0
Riti Verma  1 day ago
Against your knn.

Speedup marks will be relative, so don't worry.

0
Anonymous Calc 3  1 day ago
If it will be against our own knn then some might try to make their knn now fast, like make it intentionally complex and slow so that they could get higher speedup. It won't be fair.
0
Anonymous Calc 3  1 day ago
make their knn *not* fast, like they may just make it sequential.
0
Riti Verma  1 day ago
There are marks for optimization of knn as well so idk how that's going to help their case if they deliberately write a slow code. But considering this point, we will evaluate both approx knn and exact knn against our sequential exact knn code only.
3

Unresolved
Rachit Bhalani  1 day ago
The architecture of the GPU will be fixed right? (32 threads per warp, 192 KB of shared memory/L1 cache , etc)
1

Unresolved
Anonymous Poet 3  1 day ago
while allocating value of 'n' itself from the input file

cudaMalloc(&sx,k*sizeof(int)), i get this error-
./a2 input.txt knn
CUDA error at a2.cu:332 -> out of memory

is this an error because of HPC?

1

Resolved
Eeshan Yadav  1 day ago
can we assume n>=k+1 always?
0
Riti Verma  1 day ago
yes
0

Resolved
Anonymous Atom 3  1 day ago
If, during iterations, one or more of the clusters do not get any closest points (one or more of the clusters become empty). Then, how should the new centroids be computed?
0
Anonymous Atom 3  14 hours ago
I mean to ask if my centroids were [p1, p2, p3] and in the next iteration, I find that all the points are closest to p2 or p3 (no points in the cluster of p1). In the next iteration, when we are updating the centroids by taking sum of points / number of points for each cluster, how do we change the centroid of cluster 1?
0
Riti Verma  8 hours ago
leave it unchanged.
0

Resolved
Anonymous Comp 3  16 hours ago
MAE loss will be compared between which 2 files for approx K-Means ? Correct K-Means vs Approx K-Means 

Or

Students Approx K-Means vs TA's Approx K-Means ?

1
Riti Verma  8 hours ago
There's no approx kmeans.

Approx knn mae loss will be computed between correct exact knn and approx knn.

0

Resolved
Anonymous Gear 4  14 hours ago
Will our code be run on multiple gpu too or just one gpu?
what about cpus, will they be varied?
0
Riti Verma  8 hours ago
they will be varied.
0

Resolved
Anonymous Mouse 4  13 hours ago
Is there any range for the values of x[i] , y[i] , z[i] the 3d coordinates?
0
Anonymous Atom 3  10 hours ago
I think it was answered here https://piazza.com/class/mk2exak3kau4fk/post/26_f5
1
Endorsed by Instructor (Riti Verma)

Resolved
Anonymous Scale 4  11 hours ago
Will file IO be included in the time measurement?
0
Riti Verma  8 hours ago
yes.
0

Resolved
Anonymous Beaker 4  10 hours ago
The generator file produces integer values for coordinate inputs, while the PDF specification states that points will be from ℝ³ (real numbers). When points are treated as floating-point values, floating-point precision errors can cause intensity mismatches — specifically, a ±1 discrepancy when converting intensities to uint8.

Will the input coordinates be provided as integers or floating-point numbers?

Acceptable Tolerance Range (if floating-point) : If inputs are provided in floating-point format, will there be an acceptable tolerance range within which the outputs listed below will be considered correct?

2
Anonymous Beaker 3  10 hours ago
I believe this was answered here:


https://piazza.com/class/mk2exak3kau4fk/post/26_f17

0
Anonymous Beaker 4  10 hours ago
ok thanx. I missed this.

0

Unresolved
Anonymous Poet 4  9 hours ago
how much ram can we use?
also what are the min,max number of cpu cores that will be used to test?
0

Resolved
Anonymous Calc 4  8 hours ago
just to confirm. there is no variable with float datatype right? everything is done in integer space.
0
Riti Verma  7 hours ago
only the new intensity before taking the floor value.
0

Resolved
Anonymous Atom 4  8 hours ago
$ qstat -f 797657.pbshpc | grep -iE "comment|reason|state"
    job_state = Q
    substate = 10
    comment = Not Running: Insufficient amount of resource: ngpus (R: 1 A: 0 T:

my job hasnt started since past 30 minutes or more. I cant test the code on hpc since no gpu is available for use right now. what should i do in this case ? the deadline is tomorrow and it isnt my fault that my job isnt running. 

1
Riti Verma  7 hours ago
that is why you were advised to start early. a lot of people might be requesting the jobs right now so it will likely take time. try with lesser number of cores.
0
Anonymous Atom 4  7 hours ago
its been more than an hour now, still no resource has been freed. also i have requested for 1 cpu and 1 gpu only even then this is happening.
3
Anonymous Atom 4  7 hours ago
nevermind, finally got the job started after almost 1.5 hrs
0
Anonymous Comp  6 hours ago
Same issue waiting from 30 mins still job has not started
2

Resolved
Anonymous Comp  6 hours ago
When we output new intensity values do we output them as floats or integers?
0
Riti Verma  6 hours ago
integers
1

Resolved
Anonymous Comp 3  6 hours ago
To run the code on HPC, we need to load appropriate cuda version, say 11.4. Do we have to include to load the correct version in our makefile or the environment setup by the TAs ensure it is available ?
0
Riti Verma  5 hours ago
module load compiler/gcc/9.1.0
module load suite/nvidia-hpc-sdk/20.7/cuda11.0
these modules will be loaded by us.

1

Resolved
Anonymous Helix 4  5 hours ago
Do we also have to submit our sequential version of KNN or just the cuda ones ?
0
Riti Verma  4 hours ago
just cuda.
0

Resolved
Anonymous Gear 5  4 hours ago
It says waiting for job ..... to start and stayed like that for 30 minutes or so and still doesnt proceed further how will i test my code , is this a sign of credits being comsumed fully i dont know please specify what should i do
0
Riti Verma  4 hours ago
amgr checkbalance project -n col7880.<kerberos>.course
you can check your credits left from this.

0
Anonymous Comp  2 hours ago
running it shows cost, what is the credit limit ?
1

Resolved
Anonymous Helix 4  3 hours ago
Could you clarify whether the three tasks will be benchmarked in separate runs or in a single execution? This affects whether shared preprocessing/device-memory reuse across tasks is meaningful for optimization.
1
Riti Verma  2 hours ago
separate
0

Unresolved
Rachit Bhalani  3 hours ago
HPC is taking a lot of time in offering a job; I've been waiting since 1 hour for the same.
Kindly inform how I should test my code
7
Anonymous Mouse 5  2 hours ago
I think you can try on colab, it has cuda and openmp support. I am also trying on that
2

Unresolved
Anonymous Scale 5  2 hours ago
The HPC is taking a lot of time to allocate resources. Each iteration takes around an hour which makes it extremely difficult to check out code. Kindly extend. the submission deadlines and help in resolving this issue. Lot of us do not have access to CUDA resources apart from HPC.

4
Vanshika  2 hours ago
Yes, same issue. It's more than an hour since I have been waiting.
3

Unresolved
Vanshika  2 hours ago
[cs1230746@login07 ~]

$ qsub -I -P col7880.cs1230746.course -N test -l select=1:ncpus=1:ngpus=1

qsub: waiting for job 798566.pbshpc to start

qsub: job 798566.pbshpc apparently deleted


Now, this is happening again and again.

7

Resolved
Anonymous Beaker 5  1 hour ago
My queued jobs are getting deleted on HPC. I am not able to check my credits also, although I am sure they wouldn't have expired. Please look into this.
2
Anonymous Beaker 5  2 hours ago
nvm it ran after multiple retries