#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>


bool* create_fetures(bool* x_features, int* x, long num_record, int max_len) {
	for (long record_idx = 0; record_idx < num_record; record_idx ++) {
		long index_shift = record_idx * max_len * 10;
		int digit_idx = max_len - 1;
		int n = x[record_idx];
		do {
			int digit = n % 10;
			x_features[index_shift + digit_idx*10 + digit] = 1;
			digit_idx -= 1;
		} while (n /= 10);
		
	}
	return x_features;
}


struct create_feature_params_structure{
    //Or whatever information that you need
    bool *x_features;
    int *x;
	int record_start;
	int record_end; 
	int max_len;
};

void *create_fetures_mutlt_thread(void *args) {
	struct create_feature_params_structure *actual_args = args;
	bool* x_features = actual_args->x_features;
	int* x = actual_args->x;
	int record_start = actual_args->record_start;
	int record_end = actual_args->record_end;
	long max_len = actual_args->max_len;

	for (long record_idx = record_start; record_idx < record_end; record_idx ++) {
		long index_shift = record_idx * max_len * 10;
		int digit_idx = max_len - 1;
		int n = x[record_idx];
		do {
			int digit = n % 10;
			x_features[index_shift + digit_idx*10 + digit] = 1;
			digit_idx -= 1;
		} while (n /= 10);
		
	}
	free(actual_args);
	return NULL;
}


bool* create_fetures_mutlt_thread_mgr(bool* x_features, int* x, long num_record, int max_len, int num_thread) {

	pthread_t threads[num_thread];

	// printf("Multi-threading manager: creating %d threads \n", num_thread);
	long size_partition = (long)(num_record / num_thread) + 1;
	int rc;
	for (int thread_id = 0; thread_id < num_thread; thread_id++) {
		int record_start = size_partition * thread_id;
		int record_end = size_partition * (thread_id + 1);
		if (thread_id == num_thread -1) {
			record_end = num_record;
		}
		// printf("creating: %d thread\n", thread_id);

		struct create_feature_params_structure *args;
		args = malloc(sizeof(*args));
		args->x_features = x_features;
		args->x = x;
		args->record_start = record_start;
		args->record_end = record_end;
		args->max_len = max_len;

		rc = pthread_create(&threads[thread_id], NULL, create_fetures_mutlt_thread, args);
		if (rc) {
        //  printf("Error:unable to create thread, %d\n", rc);
         exit(-1);
      }
	}

	for (int thread_id = 0; thread_id < num_thread; thread_id++) {
		pthread_join(threads[thread_id], NULL);
	}

	// pthread_exit(NULL);
	return x_features;
}

#define randnum(min, max) \
        ((rand() % (int)(((max) + 1) - (min))) + (min))


void test_single_thread() {
	long num_record = 399999999;
	long size1 = num_record*8*10;
	bool *x_f_arr = calloc(size1, sizeof(bool));
	int *x_data = calloc(num_record, sizeof(int));

	for (int i = 0; i < num_record; i++) {
		int random_num = randnum(0, 74999999);
		x_data[i] = random_num;
	}
	create_fetures(x_f_arr, x_data, num_record, 8);

	for (int i = 0; i < 80; i++) {
		if (i % 80 == 0) {
			printf("||\n");
		}
		if (i % 10 == 0) {
			printf("|");
		}
		printf("%d,", x_f_arr[i]);
	}
	printf("Value : %d\n", x_data[0]);
}


void test_multi_thread() {
	long num_record = 399999999;
	long size1 = num_record*8*10;
	bool *x_f_arr = calloc(size1, sizeof(size_t));
	int *x_data = calloc(num_record, sizeof(size_t));

	for (int i = 0; i < num_record; i++) {
		int random_num = randnum(0, 74999999);
		x_data[i] = random_num;
	}
	create_fetures_mutlt_thread_mgr(x_f_arr, x_data, num_record, 8, 4);

	for (int i = 0; i < 80; i++) {
		if (i % 80 == 0) {
			printf("||\n");
		}
		if (i % 10 == 0) {
			printf("|");
		}
		printf("%d,", x_f_arr[i]);
	}
	printf("Value : %d\n", x_data[0]);
}

long aux_look_up(int* x, int val, long num_record) {
	for (long record_idx = 0; record_idx < num_record; record_idx ++) {
		if (x[record_idx] == val) {
			return record_idx;
		}
	}
	return -1;
}


long aux_look_up_bin(int* x, int val, long num_record) {
	long low = 0;
	long high = num_record - 1;

	while (low <= high) {

		// Finding the mid using floor division
		long mid = low + ((high - low) / 2);

		// Target value is present at the middle of the array
		if (x[mid] == val) {
			return mid;
		}

		// Target value is present in the low subarray
		else if (val < x[mid]) {
			high = mid - 1;
		}

		// Target value is present in the high subarray
		else if (val > x[mid]) {
			low = mid + 1;
		}
	}
	return -1;
}

void test_look_up() {
	long num_record = 399999999;
	int *x_data = calloc(num_record, sizeof(size_t));

	for (int i = 0; i < num_record; i++) {
		int random_num = randnum(0, 74999999);
		x_data[i] = random_num;
	}
	
	// long result = aux_look_up(x_data, 76999999, num_record);
	long result = aux_look_up_bin(x_data, 76999999, num_record);
	printf("Value Index : %ld\n", result);
}

int main() {
	// Test for up to 30 GB array without issue 
	// test_single_thread();
	// test_multi_thread();
	test_look_up();
}