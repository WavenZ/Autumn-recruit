#### 1. 并查集
```cpp
int find(int x){
    return pre[x] == x ? x : pre[x] = find(pre[x]);
}
void merge(int a, int b){
    pre[find(a)] = find(b);
}
```

#### 2. 快排
```cpp
int partition(vector<int>& vec, int start, int end){
    int pos = start;
    for(int i = start; i < end; ++i){
        if(vec[i] < vec[end]) swap(vec[i], vec[pos++]);
    }
    swap(vec[pos], vec[end]);
    return pos;
}
void quick_sort(vector<int>& vec, int start, int end){
    if(start >= end) return;
    int pos = partition(vec, start, end);
    quick_sort(vec, start, pos - 1);
    quick_sort(vec, pos + 1, end);
}
```

#### 3. 插入排序
```cpp
void insert_sort(vector<int>& vec, int start, int end){
    if(start >= end) return;
    for(int i = start; i <= end; ++i){
        int curr = vec[i];
        int j = i - 1;
        for(; j >= 0 && vec[j] > curr; --j) vec[j + 1] = vec[j];
        vec[j + 1] = curr;
    }
}
```

#### 4. 选择排序
```cpp
void select_sort(vector<int>& vec, int start, int end){
    if(start >= end) return;
    for(int i = start; i < end; ++i){
        swap(vec[i], *min_element(vec.begin() + i, vec.end()));
    }
}
```

#### 5. 二分查找
```cpp
int binary_search(vector<int>& vec, int start, int end, int target){
    int low = start, high = end;
    while(low <= high){
        int mid = low + ((high - low) >> 1);
        if(vec[mid] < target) low = mid + 1;
        else if(vec[mid] == target) return mid;
        else high = mid - 1;
    }
    return -1;
}
int lower_bound(vector<int>& vec, int start, int end, int target){
    int low = start, high = end;
    while(low < high){
        int mid = low + ((high - low) >> 1);
        if(vec[mid] < target) low = mid + 1;
        else high = mid;
    }
    return high;
}
int upper_bound(vector<int>& vec, int start, int end, int target){
    int low = start, high = end;
    while(low < high){
        int mid = (low + high + 1) >> 1;
        if(vec[mid] <= target) low = mid;
        else high = mid - 1;
    }
    return low + 1;
}
```

#### 6. 层序遍历二叉树
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

#### 7. 链表翻转
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
    return nullptr;
}
```

#### 9. 最小公倍数/最大公约数
```cpp
int gcd(int a, int b){
    return b ? gcd(b, a % b) : a;
}
int lcm(int a, int b){
    return a / gcd(a, b) * b;
}
```

#### 10. 全排列
```cpp
vector<vector<int>> ans;
void permutation(vector<int>& vec, int curr){
    if(curr == vec.size()) ans.push_back(vec);
    for(int i = curr; i < vec.size(); ++i){
        swap(vec[i], vec[curr]);
        permutation(vec, curr + 1);
        swap(vec[i], vec[curr]);
    }
}
void permutation(vector<int>& vec){
    permutation(vec, 0);
}
```
```cpp
vector<vector<int>> ans;
void permutation(vector<int>& vec, vector<int>& track, vector<bool>& vis){
    if(track.size() == vec.size()) ans.push_back(track);
    for(int i = 0; i < vec.size(); ++i){
        if(vis[i]) continue;
        track.push_back(vec[i]); vis[i] = true;
        permutation(vec, track, vis);
        track.pop_back(); vis[i] = false;
    }
}
void permutation(vector<int>& vec){
    vector<int> track;
    vector<bool> vis(vec.size(), false);
    permutation(vec, track, vis);
}
```
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
        for(; j > i; --j) if(vec[j] > vec[i]) break;
        swap(vec[i], vec[j]);
        reverse(vec.begin() + i + 1, vec.end());
        ans.push_back(vec);
    }
}
```
```cpp
vector<vector<int>> ans;
void permutation(vector<int> vec, int curr){
    if(curr == vec.size()) ans.push_back(vec);
    for(int i = curr; i < vec.size(); ++i){
        if(i != curr && vec[i] == vec[curr]) continue;
        swap(vec[i], vec[curr]);
        permutation(vec, curr + 1);
    }
}
void permutation(vector<int>& vec){
    sort(vec.begin(), vec.end());
    permutation(vec, 0);
}
```

#### 12. 子集生成
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
    return ans;
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
            // 注意：这里判断是否到达终点
            if(curr == target) return step;
            // 将 curr 的邻接节点加入队列
            for(Node x : Next[curr]){
                if(vis.count(x)) continue;
                vis[x] = 1;
                Q.push(x);
            }
        }
        step++;
    }
    return step;
}
```

#### 14. 0-1背包问题
```cpp
int bb01(vector<int>& weight, vector<int>& val, int W){
    int N = weight.size();
    vector<vector<int>> dp(N + 1, vector<int>(W + 1, 0));
    for(int i = 1; i <= N; ++i){
        for(int w = 1; w <= W; ++w){
            if(w < weight[i - 1]) dp[i][w] = dp[i - 1][w];
            else dp[i][w] = max(dp[i - 1][w],
                                dp[i - 1][w - weight[i - 1]] + val[i - 1]);
        }
    }
    return dp[N][W];
}
```

#### 15. 高效寻找素数
```cpp
int countPrime(int n){
    if(n < 3) return 0;
    vector<bool> isPrime(n, true);
    for(int i = 2; i * i <= n; ++i){
        if(isPrime[i]){
            for(int j = 2 * i; j < n; j += i){
                isPrime[j] = false;
            }
        }
    }
    return count(isPrime.begin() + 2, isPrime.end(), true);
}
```

#### 16. 递归快速幂
```cpp
int pow(int a, int b){
    if(b == 0) return 1;
    if(b & 1){
        return a * pow(a, b - 1);
    }else{
        int temp = pow(a, b / 2);
        return temp * temp;
    }
    return 0;
}
```

#### 17. 单调栈
```cpp
vector<int> monoStack(vector<int>& height){
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