// simply sort and retrieve k-1th element
#include <iostream>
#include <vector>
#include <algorithm>
int main() {
	//code
	int num_tests;
	std::cin >> num_tests;
	for (int i = 0; i < num_tests; i++)
	{
	    int num;
	    std::cin >> num;
	    std::vector<int> arr(num, 0);
	    for (int j = 0; j < num; j++)
	        std::cin >> arr[j];
	    int k;
	    std::cin >> k;
	    std::sort(arr.begin(), arr.end());
	    std::cout << arr[k - 1] << std::endl;
	}
	return 0;
}

Time complexity - O(nlogn), space complexity - O(n)


// min heap(priority_queue)
#include <iostream>
#include <vector>
#include <queue>
int main() {
	//code
	int num_tests;
	std::cin >> num_tests;
	for (int i = 0; i < num_tests; i++)
	{
	    int num;
	    std::cin >> num;
	    int temp;
	    std::priority_queue<int, std::vector<int>, std::greater<int>> pq;
	    for (int j = 0; j < num; j++)
	    {
	        std::cin >> temp;
	        pq.push(temp);
	    }
	    int k;
	    std::cin >> k;
	    for (int j = 0; j < k; j++)
	    {
	        temp = pq.top();
	        pq.pop();
	    }
	    std::cout << temp << std::endl;
	}
	return 0;
}
Time Complexity - heap creation O(n), extracting each element O(logn) so getting kth element - O(n + klogn), space complexity - O(n).



// max heap(priority_queue)
#include <iostream>
#include <vector>
#include <queue>
int main() {
	//code
	int num_tests;
	std::cin >> num_tests;
	for (int i = 0; i < num_tests; i++)
	{
	    int num;
	    std::cin >> num;
	    int temp;
	    std::priority_queue<int> pq;
	    std::vector<int> arr;
	    for (int j = 0; j < num; j++)
	    {
	        std::cin >> temp;
	        arr.push_back(temp);
	    }
	    int k;
	    std::cin >> k;
	    for (int j = 0; j < k; j++)
	        pq.push(arr[j]);
	    for (int j = k; j < num; j++)
	    {
	        pq.push(arr[j]);
	        pq.pop();
	    }
	    std::cout << pq.top() << std::endl;
	}
	return 0;
}
Time Complexity - heap creation O(k), extracting each element O(logk) so getting kth element - O(k + (n - k)logk), space complexity - O(k).


// using quick select (a variation of quick sort)
#include <iostream>
#include <vector>

int partition(std::vector<int> &arr, int l, int r)
{
    int pivot = arr[l];
    int left = l, right = r + 1;
    while (left < right)
    {
        while ((left < r) && (arr[++left] < pivot)) ;
        while ((right > l) && (arr[--right] > pivot)) ;
        if (left < right)
            std::swap(arr[left], arr[right]);
    }
    std::swap(arr[l], arr[right]);
    return right;
}

int select (std::vector<int> &arr, int n, int k)
{
    int left = 0, right = n - 1;
    while (left < right)
    {
        int part = partition(arr, left, right);
        if (part == k - 1)
            break;
        else if (part > k - 1)
            right = part - 1;
        else
            left = part + 1;
    }
    return arr[k - 1];
}

int main() {
	//code
	int num_tests;
	std::cin >> num_tests;
	for (int i = 0; i < num_tests; i++)
	{
	    int num;
	    std::cin >> num;
	    int temp;
	    std::vector<int> arr(num, 0);
	    for (int j = 0; j < num; j++)
	    {
	        std::cin >> temp;
	        arr[j] = temp;
	    }
	    int k;
	    std::cin >> k;
	    int res = select(arr, num, k);
	    std::cout << res << std::endl;
	}

	return 0;
}

// Time Complexity - The worst case time complexity of this method is O(n^2), but it works in O(n) on average.


// Quick Select modified (Expected Linear Time) - Randomized QuickSelect
#include <iostream>
#include <vector>

int partition(std::vector<int> &arr, int l, int r)
{
    int n = r -l + 1;
    int rand_index = rand() % n;
    std::swap(arr[l + rand_index], arr[l]);
    int pivot = arr[l];
    int left = l, right = r + 1;
    while (left < right)
    {
        while ((left < r) && (arr[++left] < pivot)) ;
        while ((right > l) && (arr[--right] > pivot)) ;
        if (left < right)
            std::swap(arr[left], arr[right]);
    }
    std::swap(arr[l], arr[right]);
    return right;
}

int select (std::vector<int> &arr, int n, int k)
{
    int left = 0, right = n - 1;
    while (left < right)
    {
        int part = partition(arr, left, right);
        if (part == k - 1)
            break;
        else if (part > k - 1)
            right = part - 1;
        else
            left = part + 1;
    }
    return arr[k - 1];
}

int main() {
	//code
	int num_tests;
	std::cin >> num_tests;
	for (int i = 0; i < num_tests; i++)
	{
	    int num;
	    std::cin >> num;
	    int temp;
	    std::vector<int> arr(num, 0);
	    for (int j = 0; j < num; j++)
	    {
	        std::cin >> temp;
	        arr[j] = temp;
	    }
	    int k;
	    std::cin >> k;
	    int res = select(arr, num, k);
	    std::cout << res << std::endl;
	}

	return 0;
}

/*
 *The worst case time complexity of the above solution is still O(n2). In worst case, the randomized function may always pick a corner element. The expected time complexity of above randomized QuickSelect is Θ(n).
 *
 */


// Worst Case linear time algorithm  - 
// The idea in this new method is similar to quickSelect(), we get worst case linear time by selecting a pivot that divides array in a balanced way         (there are not very few elements on one side and many on other side).
#include <iostream>
#include <vector>
#include <algorithm>


int findMedian(std::vector<int> arr, int start, int n)
{
    std::sort(arr.begin() + start, arr.begin() + start + n);
    return *(arr.begin() + start + n/2);
}

int partition(std::vector<int> &arr, int l, int r, int midOfMed)
{
    int i;
    //std::cout << midOfMed << " " << "foo" << std::endl;
    for (i = l; i < r; i++)
        if (arr[i] == midOfMed)
            break;
    std::cout << midOfMed << std::endl;
    for (int val: arr)
        std::cout << val << " ";
    std::cout << std::endl;
    std::swap(arr[i], arr[l]);
    for (int val: arr)
        std::cout << val << " ";
    std::cout << std::endl;
    int pivot = arr[l];
    int left = l, right = r + 1;
    while (left < right)
    {
        while ((left < r) && (arr[++left] < pivot));
        while ((right > l) && (arr[--right] > pivot));
        if (left < right)
            std::swap(arr[left], arr[right]);
    }
    std::swap(arr[l], arr[right]);
    return right;
}

int kthSmallest(std::vector<int> &arr, int l, int r, int k)
{
    if (k > 0 && k <= r - l + 1)
    {
        int n = r - l + 1;
        std::vector<int> median((n + 4) / 5, 0);
        int i = 0;
        for (; i < n / 5; i++)
            median[i] = findMedian(arr, i*5, 5);
        
        if (i*5 < n)
        {
            median[i] = findMedian(arr, i*5, n % 5);
            i++;
        }
        //std::cout << "foo" << std::endl;
        //for (int val: median)
        //    std::cout << val << std::endl;
        int medOfMed = (i == 1) ? median[i - 1] : kthSmallest(median, 0, i - 1, i / 2);
        int pos = partition(arr, l, r, medOfMed);
        std::cout << "medOfMed: " << medOfMed << " pos: " << pos  << " l:" << l << " r:" << r << std::endl;
        if (pos == k - 1)
            return medOfMed;
        else if (pos > k - 1)
            return kthSmallest(arr, l, pos - 1, k);
        else
            return kthSmallest(arr, pos + 1, r, k);
    }
}
int main() {
	//code
	int num_tests;
	std::cin >> num_tests;
	for (int i = 0; i < num_tests; i++)
	{
	    int num;
	    std::cin >> num;
	    std::vector<int> arr(num, 0);
	    for (int j = 0; j < num; j++)
	        std::cin >> arr[j];
	    int k;
	    std::cin >> k;
	    int res = kthSmallest(arr, 0, num - 1, k);
	    std::cout << res << std::endl;
	}
	return 0;
}

// Above code doesn't work, so needs debugging.
