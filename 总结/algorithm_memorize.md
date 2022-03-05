#### 1. 并查集
```cpp
int pre[MAXN];
void init(int n) {
    for (int i = 0; i < n; ++i)
        pre[i] = i;
}
int find(int x) { 
    // pre[x] == x indicates that x is a root node.
    // pre[x] = find(pre[x]) just do the path compression.
    return pre[x] == x ? x : pre[x] = find(pre[x]); 
}
void merge(int a, int b){
    // Merge two sets.
    pre[find(a)] = find(b);
}
```
#### 2. 快排
稳定性：不稳定
时间复杂度：`O(n2)/O(nlgn)`，空间复杂度：`O(1)`
```cpp
int partition(int* vec, int start, int end){
    int pos = start;
    for(int i = start; i < end; ++i){
        if(vec[i] < vec[end]) swap(vec[i], vec[pos++]);
    }
    swap(vec[pos], vec[end]);
    return pos;
}
void quick_sort(int* vec, int start, int end){
    if(start >= end) return;
    int pos = partition(vec, start, end);
    quick_sort(vec, start, pos - 1);
    quick_sort(vec, pos + 1, end);
}
```
#### 3. 插入排序
稳定性：稳定
时间复杂度：`O(n2)`，空间复杂度：`O(1)`
```cpp
void insert_sort(vector<int>& vec, int start, int end){
    if(start >= end) return;
    int n = end - start + 1;
    for(int pos = 0; pos < n; ++pos){
        int curr = vec[pos];
        int i = pos - 1;
        for(; i >= 0 && vec[i] > curr; --i){
            vec[i + 1] = vec[i];
        }
        vec[i + 1] = curr;
    }
}
```
#### 4. 选择排序
稳定性：不稳定
时间复杂度：`O(n2)`，空间复杂度：`O(1)`
```cpp
void select_sort(int* vec, int start, int end){
    if(start >= end) return;
    for(int i = start; i < end; ++i){
        swap(vec[i], *min_element(vec + i, vec + end + 1));
    }
}
```
#### 5. 二分查找
时间复杂度：`O(lgn)`，空间复杂度：`O(1)`
```cpp
int binary_search(int* vec, int start, int end, int target){
    int lo = start, hi = end;
    while(lo <= hi){
        int mid = lo + ((hi - lo) >> 1);
        if(vec[mid] < target) lo = mid + 1;
        else if(vec[mid] == target) return mid;
        else hi = mid - 1;
    }
    return -1;
}
int lower_bound(int* vec, int start, int end, int target){
    int lo = start, hi = end;
    while(lo < hi){
        int mid = lo + ((hi - lo) >> 1);
        if(vec[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return hi;
}
int upper_bound(int* vec, int start, int end, int target){
    int lo = start, hi = end;
    while(lo < hi){
        int mid = (lo + hi + 1) >> 1;
        if(vec[mid] <= target) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}
```
#### 6. 层序遍历二叉树（BFS） 
```cpp
queue<TreeNode*> Q;
if(root) Q.push(root);
while(!Q.empty()){
    for(int i = Q.size(); i > 0; --i){
        TreeNode* curr = Q.front(); Q.pop();
        ... // 处理当前节点
        if(curr->left) Q.push(curr->left);
        if(curr->right) Q.push(curr->right);
    }
}
```
#### 7. 链表反转
迭代三指针法
```cpp
Node* list_reverse(Node* head){
    if(head == nullptr || head->next == nullptr) return head;
    Node* prev = nullptr, *curr = head, *next = head->next;
    while(curr){
        curr->next = prev;
        prev = curr;
        curr = next;
        if(next) next = next->next;
    }
    return prev;
}
```
#### 8. 有序链表合并
递归法
```cpp
Node* list_merge(Node* l1, Node* l2){
    if(l1 == nullptr) return l2;
    if(l2 == nullptr) return l1;
    if(l1->val <= l2->val){
        l1->next = list_merge(l1->next, l2);
        return l1;
    }else{
        l2->next = list_merge(l1, l2->next);
        return l2;
    }
    return l1;
}
```

#### 9. 最小公倍数/最大公约数

辗转相除法

```cpp
int gcd(int a, int b){  // 最大公约数
    return b ? gcd(b, a % b) : a;
}
int lcm(int a, int b){
    return a / gcd(a, b) * b;
}
```

#### 10. 全排列

递归解法1：
```cpp
vector<vector<int>> ans;
void permutation(vector<int>& vec, int curr){
    if(curr == vec.size()){
        ans.push_back(vec);
    }
    for(int i = curr; i < vec.size(); ++i){
        swap(vec[curr], vec[i]);
        permutation(vec, curr + 1);
        swap(vec[curr], vec[i]);
    }
}
void permutation(vector<int>& vec){
    permutation(vec, 0);
}
```
递归解法2：
```cpp
vector<vector<int>> ans;
void permutation(vector<int>& vec, vector<int>& track, vector<bool>& vis){
    if(track.size() == vec.size()){
        ans.push_back(track);
    }
    for(int i = 0; i < vec.size(); ++i){
        if(vis[i]) continue;
        track.push_back(vec[i]); vis[i] = true;
        permutation(vec, i + 1, track);
        track.pop_back(); vis[i] = false;
    }
}
void permutation(vector<int>& vec){
    vector<int> track;
    vector<bool> vis(vec.size(), false);
    permutation(vec, track, vis);
}
```


非递归解法：
1. 从右向左，找到第一个满足`vec[i] < vec[i + 1]`的位置`i`。
2. 从右向左，找到比`vec[i]`大的数中最小的一个`vec[j]`的位置`j`。
3. 交换`vec[i]`和`vec[j]`。
4. 翻转`vec[i + 1 : end]`。
```cpp
vector<vector<int>> ans;
void permutation(vector<int>& vec){
    ans.push_back(vec);
    if(vec.size() < 2) return;
    while(1){
        int i = vec.size() - 2;
        for(; i >= 0; --i) if(vec[i] < vec[i + 1]) break;
        if(i == -1){
            reverse(vec.begin(), vec.end()); break;
        }
        int j = vec.size() - 1;
        for(; j > i; --j) if(vec[i] < vec[j]) break;
        swap(vec[i], vec[j]);
        reverse(vec.begin() + i + 1, vec.end());
        ans.push_back(vec);
    }
}
```

含重复元素的全排列：
```cpp
vector<vector<int>> ans;
void permutation(vector<int> vec, int curr){
    if(curr == vec.size()){
        ans.push_back(vec);
    }
    for(int i = curr; i < vec.size(); ++i){
        if(i != curr && vec[i] == vec[curr]) continue;
        swap(vec[curr], vec[i]);
        permutation(vec, curr + 1);
        // swap(vec[start], vec[i]);
    }
}
void permutation(vector<int>& vec){
    sort(vec.begin(), vec.end()); // *
    permutation(vec, 0);
}
```
其中，`vec`是按值传递的方式进行传参的。

#### 11. 子集生成

利用回溯法可以完成不含重复元素的子集生成

```cpp
vector<vector<int>> ans;
void backtrack(vector<int>& vec, int curr, vector<int>& track){
    ans.push_back(track);
    for(int i = curr; i < vec.size(); ++i){
        track.push_back(vec[i]);
        backtrack(vec, i + 1, track);
        track.pop_back();
    }
}
void subSets(vector<int>& vec){
    vector<int> track;
    backtrack(vec, 0, track);
}
```
#### 13. BFS框架
```cpp
int BFS(Node start, Node target){
    queue<Node> Q;
    unordered_map<Node, int> vis;
    Q.push(start);
    int step = 0;
    while(!Q.empty()){
        for(int i = Q.size(); i > 0; --i){
            Node curr = Q.front(); Q.pop();
            // 注意：这里判断是否到终点
            if(curr == target) return step;
            // 将 curr 的邻接节点加入队列
            for(Node x : Next[curr]){
                if(vis.count(x)) continue;
                vis[x] = 1;
                Q.push(x);
            }
        }
        step++; // 这里更新步数
    }
    return step;
}
```

#### 14. 0-1背包问题

给你一个可装载重量为`W`的背包和`N`个物品，每个物品有重量和价值两个属性。其中第`i`个物品的重量为`weight[i]`，价值为`val[i]`，现在用这个背包装物品，最多能装的价值是多少？

`dp[i][w]`的定义如下：对于前`i`个物品，当前背包的容量为`w`，这种情况下可以装的最大价值是`dp[i][w]`。

框架如下：
```cpp
int bb01(const vector<int>& weight, const vector<int>& val, int W){
    int N = weight.size();
    vector<vector<int>> dp(N + 1, vector<int>(W + 1));
    for(int i = 1; i <= N; ++i){
        for(int w = 1; w <= W; ++w){
            if(w < weight[i - 1]) dp[i][w] = dp[i - 1][w];
            else dp[i][w] = max(dp[i - 1][w],                   // 物品 i 不放入背包中 
                                dp[i - 1][w - weight[i - 1]] + val[i - 1]);  // 物品 i 放入背包中
            cout << dp[i][w] << " ";
        }
        cout << endl;

    }
    return dp[N - 1][W];
}
```

#### 15. 高效寻找素数

1. 朴素算法1
```cpp
class Solution {
public:
    int countPrimes(int n) {
        auto isPrime = [](int k) -> bool{
            for(int i = 2; i < k; ++i){
                if(k % i == 0) return false;
            }
            return true;
        };
        int ans = 0;
        for(int i = 2; i < n; ++i){
            ans += isPrime(i);
        }
        return ans;
    }
};
```
2. 朴素算法2
```cpp
class Solution {
public:
    int countPrimes(int n) {
        auto isPrime = [](int k) -> bool{
            for(int i = 2; i * i <= k; ++i){
                if(k % i == 0) return false;
            }
            return true;
        };
        int ans = 0;
        for(int i = 2; i < n; ++i){
            ans += isPrime(i);
        }
        return ans;
    }
};
```
优化点1：`isPrime`中的`for`循环不需要到`i < k`。

3. 高效算法1
```cpp
class Solution {
public:
    int countPrimes(int n) {
        if(n < 3) return 0;
        bool* isPrime = new bool[n];
        memset(isPrime, true, n * sizeof(bool));
        for(int i = 2; i < n; ++i){
            if(isPrime[i]){
                for(int j = 2 * i; j < n; j += i){
                    isPrime[j] = false;
                }
            }
        }
        int ans = count(isPrime + 2, isPrime + n, true);
        return ans;
    }
};
```
优化点2：如果`i`为素数，则`2i`、`3i` ...肯定不为素数。
4. 高效算法2
```cpp
class Solution {
public:
    int countPrimes(int n) {
        if(n < 3) return 0;
        bool* isPrime = new bool[n];
        memset(isPrime, true, n * sizeof(bool));
        for(int i = 2; i * i <= n; ++i){
            if(isPrime[i]){
                for(int j = 2 * i; j < n; j += i){
                    isPrime[j] = false;
                }
            }
        }
        int ans = count(isPrime + 2, isPrime + n, true);
        return ans;
    }
};
```
优化点3：外层只需要到`sqrt(n)`即可。

#### 16. 递归快速幂

```cpp
int pow(int a, int b){
    if(b == 0) return 1;
    if(b & 1) return a * pow(a, b - 1);
    else{
        int temp = pow(a, b / 2);
        return temp * temp;
    }
    return 0;
}
```

#### 17. 单调栈

```cpp
vector<int> monoStack(vector<int> height){
    vector<int> ans(height.size());
    stack<int> s;
    for(int i = vec.size() - 1; i >= 0; --i){
        while(!s.empty() && height[s.top()] <= height[i]) s.pop();
        ans[i] = s.empty() ? 0 : s.top() - i;
        s.push(i);
    }
    return ans;
}
```



#### 
