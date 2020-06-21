#### 198. 打家劫舍
```cpp
class Solution {
public:
    int rob(vector<int>& nums) {
        if(nums.size() == 0) return 0;
        int dp0 = 0, dp1 = nums[0], temp;
        for(int i = 1; i < nums.size(); ++i){
            temp = dp0;
            dp0 = max(dp0, dp1);
            dp1 = nums[i] + temp;
        }
        return max(dp0, dp1);
    }
};
```
#### 213. 打家劫舍II
```cpp
class Solution {
public:
    int rob1(vector<int>& nums, int start, int end){
        if(start > end) return 0;
        int dp0 = 0, dp1 = nums[start], temp;
        for(int i = start + 1; i <= end; ++i){
            temp = dp0;
            dp0 = max(dp0, dp1);
            dp1 = nums[i] + temp;
        } 
        return max(dp0, dp1);
    }
    int rob(vector<int>& nums) {
        if(nums.size() == 0) return 0;
        if(nums.size() == 1) return nums[0];
        return max(rob1(nums, 0, nums.size() - 2), rob1(nums, 1, nums.size() - 1));
    }
};
```
#### 337. 打家劫舍III
```cpp
class Solution {
public:
    unordered_map<TreeNode*, int> map0, map1;
    int dfs(TreeNode* root, bool steal){
        if(root == nullptr) return 0;
        int maxn = 0;
        if(steal){
            if(map1.count(root)) return map1[root];
            maxn = max(root->val + dfs(root->left, 0) + dfs(root->right, 0), dfs(root->left, 1) + dfs(root->right, 1));
            map1[root] = maxn;
        }else{
            if(map0.count(root)) return map0[root];
            maxn = dfs(root->left, 1) + dfs(root->right, 1);
            map0[root] = maxn;
        }
        return maxn;
    }
    int rob(TreeNode* root) {
        return max(dfs(root, 0), dfs(root, 1));
    }
};
```
#### 121. 买卖股票的最佳时机

维护最低价`prices[low]`，更新最大利润`ans`。
```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if(prices.size() < 2) return 0;
        int low = 0, ans = 0;
        for(int i = 1; i < prices.size(); ++i){
            if(prices[i] < prices[low]) low = i;
            ans = max(ans, prices[i] - prices[low]);
        }
        return ans;
    }
};
```
**动态规划1**：`dp[i]`表示第`i`天买入的最大利润。
状态转换：$$
dp[i] = \begin{cases}
prices[i + 1] - prices[i] + dp[i + 1] &{dp[i + 1] > 0}
\\prices[i + 1] - prices[i] &{others}
\end{cases}
$$
也就是说，如果`dp[i + 1] > 0`（盈利），则`dp[i] = prices[i + 1] - prices[i] + dp[i + 1]`。否则，`dp[i] = prices[i + 1] - prices[i]`（第二天直接卖掉）。
```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int ans = 0;
        vector<int> dp(prices.size(), 0);
        for(int i = prices.size() - 2; i >= 0; --i){
            dp[i] = prices[i + 1] - prices[i];
            if(dp[i + 1] > 0) dp[i] += dp[i + 1];
            ans = max(ans, dp[i]);
        }
        return ans;
    }
};
```
状态压缩：由于当前状态`dp[i]`只与`dp[i + 1]`有关，因此可以进行压缩。
```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int ans = 0;
        int dp = 0, dp1 = 0;
        for(int i = prices.size() - 2; i >= 0; --i){
            dp = prices[i + 1] - prices[i];
            if(dp1 > 0) dp += dp1;
            dp1 = dp;
            ans = max(ans, dp);
        }
        return ans;
    }
};
```
**动态规划2**：`dp[i][0]`表示当前没有股票时最大利润，`dp[i][1]`表示当前有股票时的最大利润。
状态转移方程：
```cpp
dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
dp[i][1] = max(-prices[i], dp[i - 1][1]);
```
其中，
1. `dp[i][0] = ...`表示当前没有股票时的最大利润。此时，要么`i - 1`时刻也没有股票（`dp[i - 1][0]`），要么`i - 1`时刻有股票（`dp[i - 1][1]`）时现在将股票卖掉`+ prices[i]`。
2. `dp[i][1] = ...`表示当前有股票时的最大利润。此时，要么`i - 1`时刻有股票（`dp[i - 1][1]`），要么`i - 1`时刻没有股票，但现在买入一支股票（`-prices[i]`）。
```cpp
class Solution{
public:
    int maxProfit(vector<int>& prices){
        if(prices.size() == 0) return 0;
        vector<vector<int>> dp(prices.size(), vector<int>(2, 0));
        dp[0][0] = 0, dp[0][1] = -prices[0];
        for(int i = 1; i < prices.size(); ++i){
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = max(dp[i - 1][1], -prices[i]);
        }
        return max(dp[prices.size() - 1][0], dp[prices.size() - 1][1]);
    }
};
```
同样的，进行状态压缩：
```cpp
class Solution{
public:
    int maxProfit(vector<int>& prices){
        if(prices.size() == 0) return 0;
        int dp0 = 0, dp1 = -prices[0];
        for(int i = 1; i < prices.size(); ++i){
            dp0 = max(dp0, dp1 + prices[i]);
            dp1 = max(dp1, -prices[i]);
        }
        return max(dp0, dp1);
    }
};
```
#### 122. 买卖股票的最佳时机II
这道题和题121的区别在于可以多次买卖股票，因此状态转移方程需要稍加修改为如下：
```cpp
dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
dp[i][1] = max(dp[i - 1][0] - prices[i], dp[i - 1][1]);
```
其中，`dp[i][1] = ...`稍有不同，因为可以多次买卖股票。
```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if(prices.size() == 0) return 0;
        int dp0 = 0, dp1 = -prices[0];
        for(int i = 1; i < prices.size(); ++i){
            int temp0 = dp0, temp1 = dp1;
            dp0 = max(temp0, temp1 + prices[i]);
            dp1 = max(temp0 - prices[i], temp1);
        }
        return max(dp0, dp1);
    }
};
```
#### 123. 买卖股票的最佳时机III
这道题比121和122的更难一些，因为限制了买卖次数为2。此时，需要将当前的买卖次数当作一个状态，取0和1。分别表示当前时第1次买卖和第2次买卖。

1. `dp[i][0][0]`表示第`i`时刻，处于第 1 次买卖的时候，手上没有股票的状态。
2. `dp[i][0][1]`表示第`i`时刻，处于第 1 次买卖的时候，受伤有股票的状态。
3. `dp[i][1][0]`表示第`i`时刻，处于第 2 次买卖的时候，受伤没有股票的状态。
3. `dp[i][1][1]`表示第`i`时刻，处于第 2 次买卖的时候，受伤有股票的状态。

**状态转移方程**：
```cpp
dp[i][0][0] = max(dp[i - 1][0][0], dp[i - 1][0][1] + prices[i]);
dp[i][0][1] = max(dp[i - 1][0][1], -prices[i]);

dp[i][1][0] = max(dp[i - 1][1][0], dp[i - 1][1][1] + prices[i]);
dp[i][1][1] = max(dp[i - 1][1][1], dp[i - 1][i][0] - prices[i]);
```
初始状态：
```cpp
dp[0][0][1] = dp[0][1][1] = -prices[0];
```
其中，`dp[i][0][1] = max(dp[i - 1][0][1], -prices[i])`和题121相同，相当于只能进行一次买卖。
```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if(prices.size() == 0) return 0;
        vector<vector<vector<int>>> dp(prices.size(), vector<vector<int>>(2, vector<int>(2, 0)));
        dp[0][0][1] = dp[0][1][1] = -prices[0];
        for(int i = 1; i < prices.size(); ++i){
            dp[i][0][0] = max(dp[i - 1][0][0], dp[i - 1][0][1] + prices[i]);
            dp[i][0][1] = max(dp[i - 1][0][1], - prices[i]);
  
            dp[i][1][0] = max(dp[i - 1][1][0], dp[i - 1][1][1] + prices[i]);
            dp[i][1][1] = max(dp[i - 1][1][1], dp[i - 1][0][0] - prices[i]);
        }
        return dp[prices.size() - 1][1][0];
    }
};
```
经过状态压缩后：
```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if(prices.size() == 0) return 0;
        int dp00 = 0, dp01 = -prices[0], dp10 = 0, dp11 = -prices[0];
        for(int i = 1; i < prices.size(); ++i){
            int temp00 = dp00, temp01 = dp01, temp10 = dp10, temp11 = dp11;
            dp00 = max(temp00, temp01 + prices[i]);
            dp01 = max(temp01, - prices[i]);
            dp10 = max(temp10, temp11 + prices[i]);
            dp11 = max(temp11, temp00 - prices[i]);
        }
        return dp10;
    }
};
```
#### 188. 买卖股票的最佳时机 IV
这道题将能够买卖的次数作为可变值`k`给出，因此只需要将上述代码中的买卖次数状态由`0、1`修改为`0、1..k - 1`即可。

下面是压缩状态之前的代码：
```cpp
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        if(prices.size() == 0 || k == 0) return 0;
        vector<vector<vector<int>>> dp(prices.size(), vector<vector<int>>(k, vector<int>(2, 0)));
        for(int i = 0; i < k; ++i) dp[0][i][1] = -prices[0];
        vector<int> temp(k, 0), temp1(k, 0);
        for(int i = 1; i < prices.size(); ++i){
            dp[i][0][0] = max(dp[i - 1][0][0], dp[i - 1][0][1] + prices[i]);
            dp[i][0][1] = max(dp[i - 1][0][1], -prices[i]);
            for(int j = 1; j < k; ++j){
                dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i]);
                dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i]);
            }
        }
        return dp[prices.size() - 1][k - 1][0];
    }
};
```
可以看到，上述代码和前面几道题的代码差不多。进行状态压缩之后：
```cpp
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        if(prices.size() == 0 || k == 0) return 0;
        vector<int> dp(k, 0), dp1(k, 0);
        vector<int> temp(k, 0), temp1(k, 0);
        for(int i = 0; i < k; ++i) dp1[i] = -prices[0];
        for(int i = 1; i < prices.size(); ++i){
            for(int j = 0; j < k; ++j){
                temp[j] = dp[j]; temp1[j] = dp1[j];
            }
            dp[0] = max(temp[0], temp1[0] + prices[i]);
            dp1[0] = max(temp1[0], -prices[i]);
            for(int j = 1; j < k; ++j){
                dp[j] = max(temp[j], temp1[j] + prices[i]);
                dp1[j] = max(temp1[j], temp[j - 1] - prices[i]);
            }
        }
        return dp[k - 1];
    }
};
```
此外，当`k`取值很大时，程序可能会超时。而当`k`大于等于`prices.size() / 2`时，结果和不限制买卖次数是一样的，因此当`k`过大时，直接退化为题122的解法：
```cpp
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        if(prices.size() == 0 || k == 0) return 0;
        if(k >= prices.size()){ // 退化为不限制次数
            int dp = 0, dp1 = -prices[0];
            for(int i = 1; i < prices.size(); ++i){
                int temp = dp, temp1 = dp1;
                dp = max(temp, temp1 + prices[i]);
                dp1 = max(temp1, temp - prices[i]);
            }
            return dp;
        }
        vector<int> dp(k, 0), dp1(k, 0);
        vector<int> temp(k, 0), temp1(k, 0);
        for(int i = 0; i < k; ++i) dp1[i] = -prices[0];
        for(int i = 1; i < prices.size(); ++i){
            for(int j = 0; j < k; ++j){
                temp[j] = dp[j]; temp1[j] = dp1[j];
            }
            dp[0] = max(temp[0], temp1[0] + prices[i]);
            dp1[0] = max(temp1[0], -prices[i]);
            for(int j = 1; j < k; ++j){
                dp[j] = max(temp[j], temp1[j] + prices[i]);
                dp1[j] = max(temp1[j], temp[j - 1] - prices[i]);
            }
        }
        return dp[k - 1];
    }
};
```
至此，对于买卖次数为`1`次、`2`次、`k`次、无限次的情况都已经用一套模板解决。

#### 309. 最佳买卖股票时机含冷冻期

这个题也是买卖股票系列题目，买卖次数限制未无限次，但加入了另一个限制条件：卖出股票后，第二天无法购入股票。

下面是没有冷冻期时的状态转移方程：

```cpp
dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
```

增加一个冷冻状态：

```cpp
dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
dp[i][1] = max(dp[i - 1][1], dp[i - 1][2] - prices[i]);
dp[i][2] = dp[i - 1][0];
```
其中，`dp[i][2]`为冷冻状态，相当于保持前一天的无股票状态。
```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if(prices.size() == 0) return 0;
        int dp = 0, dp1 = -prices[0], dp2 = 0;
        for(int i = 1; i < prices.size(); ++i){
            int temp = dp, temp1 = dp1, temp2 = dp2;
            dp = max(temp, temp1 + prices[i]);
            dp1 = max(temp1, temp2 - prices[i]);
            dp2 = temp;
        }
        return dp;
    }
};  
```
#### 714. 买卖股票的最佳时机含手续费

这题也是买卖股票系列，不过在无限次买卖的基础上加了一个买卖的手续费`fee`。因此，每次卖出股票时，我们需要扣除`fee`的手续费，状态转移方程：
```cpp
dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee);
dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
```
下面时AC代码：
```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices, int fee) {
        if(prices.size() == 0) return 0;
        int dp = 0, dp1 = -prices[0];
        for(int i = 1; i < prices.size(); ++i){
            int temp = dp, temp1 = dp1;
            dp = max(temp, temp1 + prices[i] - fee);
            dp1 = max(temp1, temp - prices[i]);
        }
        return dp;
    }
};
```
可以看到，几乎和题122的代码一样，只是在出售的时候加上了手续费。

#### 739. 每日温度
我一开始用的是`dp`，如下：
```cpp
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& T) {
        if(T.size() == 0) return {};
        vector<int> dp(T.size(), 0);
        for(int i = T.size() - 2; i >= 0; --i){
            int pos = i + 1;
            while(1){
                if(T[i] < T[pos]){
                    dp[i] = pos - i; break;
                }
                if(dp[pos] == 0) break;
                pos += dp[pos];
            }
        }
        return dp;
    }
};
```
官方题解用的单调栈：
```cpp
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& T) {
        stack<int> S;
        vector<int> res(T.size(), 0);
        for(int i = 0; i < T.size(); i++){
            while(!S.empty() && T[S.top()] < T[i]){
                res[S.top()] = i - S.top();
                S.pop();
            }
            S.push(i);
        }
        return res;
    }
};
```
其中，栈`S`中保存的是温度单调递减的时刻。

看起来`dp`和单调栈的时间复杂度是一样的，但是`dp`方法少了一个数组，因此访存的次数会少一些，使得`dp`方法在线上的时间更快一些。

#### 127 单词接龙

这个题有一个核心点：如何快速找到某个单词 A 的所有邻接点。

当单词数量较少时，可以通过双重for循环直接暴力搜索即可：
```cpp
vector<vector<string>> (wordList.size());
for(int i = 0; i < wordList.size(); ++i){
    for(int j = i + 1; j < wordList.size(); ++j){
        if(check(wordList[i], wordList[j])){
            Next[i].push_back(wordList[j]);
            Next[j].push_back(wordLsit[i]);
        }
    }
}
```
该方法的时间复杂度是`O(n2)`，`n`为单词总个数。当`n`很大时，利用上述方式来建图会导致程序超时。

另一种方法步骤如下：
1. 将所有单词放进一个`hash`表。
2. 对于某个单词`word`，将其所有位分别改为`'a' ~ 'z'`，然后通过查询`hash`表来找到其所有邻接点。
```cpp
unordered_map<string, int> map;
for(int i = 0; i < wordList.size(); ++i) map[wordList[i]] = i;
vector<vector<string>> Next(wordList.size());
for(int i = 0; i < wordList.size(); ++i){
    string& word = wordList[word];
    for(int j = 0; j < word.size(); ++j){
        string temp = word;
        for(char ch = 'a'; ch <= 'z'; ++ch){
            temp[j] = ch;
            if(map.count(temp)) Next[i].push_back(temp);
        }
    }
}
```
此方法的时间复杂度为`O(n * wordLen)`。

下面是`AC`代码：
```cpp
class Solution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        wordList.push_back(beginWord);
        unordered_map<string, int> map;
        unordered_map<string, bool> vis;
        for(int i = 0; i < wordList.size(); ++i){
            map[wordList[i]] = i;
            vis[wordList[i]] = false;
        }
        if(!map.count(endWord)) return 0;
        int end = map[endWord];
        vector<vector<string>> Next(wordList.size());
        for(int i = 0; i < wordList.size(); ++i){
            string word = wordList[i];
            for(int j = 0; j < word.size(); ++j){
                string temp = word;
                for(char ch = 'a'; ch <= 'z'; ++ch){
                    temp[j] = ch;
                    if(map.count(temp)) Next[i].push_back(temp);
                }
            }
        }
        int len = 0;
        queue<string> Q; Q.push(beginWord);
        while(!Q.empty()){
            len++;
            for(int i = Q.size(); i > 0; --i){
                int curr = map[Q.front()]; Q.pop();
                if(curr == end) return len;
                for(int j = 0; j < Next[curr].size(); ++j){
                    if(vis[Next[curr][j]]) continue;
                    vis[Next[curr][j]] = true;
                    Q.push(Next[curr][j]);
                }
            }
        }
        return 0;
    }
};
```

#### 126 单词接龙II
`BFS` + 回溯
```cpp
class Solution {
public:
    bool check(const string& s1, const string& s2){
        if(s1.size() != s2.size()) return false;
        int cnt = 0;
        for(int i = 0; i < s1.size(); ++i){
            if(s1[i] == s2[i]) cnt++;
        }
        return cnt == (s1.size() - 1);
    }
    vector<vector<string>> ans;
    void dfs(vector<string>& wordList, vector<vector<int>>& Prev, int curr, int end, vector<string>& path, int depth){
        if(depth < 0) return;
        if(curr == end){
            ans.push_back(path); return;
        }
        for(auto& prev : Prev[curr]){
            path.push_back(wordList[prev]);
            dfs(wordList, Prev, prev, end, path, depth - 1);
            path.pop_back();
        }
    }

    vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList) {
        wordList.push_back(beginWord);
        // 找到endWord在wordList中的下标
        int end = 0;
        auto res = find(wordList.begin(), wordList.end(), endWord);
        if(res == wordList.end()) return {};
        else end = res - wordList.begin();
        // 建图
        vector<vector<int>> Next(wordList.size());
        for(int i = 0; i < wordList.size(); ++i){
            for(int j = i + 1; j < wordList.size(); ++j){
                if(check(wordList[i], wordList[j])){
                    Next[i].push_back(j);
                    Next[j].push_back(i);
                }
            }
        }

        vector<bool> vis(wordList.size());
        vector<uint8_t> dis(wordList.size(), 0xfe);
        vector<vector<int>> Prev(wordList.size());

        int len = 0;
        dis[end] = 0;
        bool finish = false;
        
        queue<int> Q;
        Q.push(end);
        while(!Q.empty() && !finish){
            for(int i = Q.size() - 1; i >= 0; --i){
                int curr = Q.front(); Q.pop();
                if(curr == wordList.size() - 1) finish = true;
                for(auto& next : Next[curr]){
                    if(dis[curr] + 1 <= dis[next]){ // 不用vis来判断是否加入到Prev中，因为可能有多条路径
                        Prev[next].push_back(curr);
                        dis[next] = dis[curr] + 1;
                        if(!vis[next]){
                            Q.push(next);
                            vis[next] = true;
                        }
                    }
                    
                }
            }
            len++;
        }
        vector<string> path;
        path.push_back(beginWord);
        dfs(wordList, Prev, wordList.size() - 1, end, path, len - 1);
        return ans;
    }
};
```

#### 322. 零钱兑换
动态规划经典题目
```cpp
class Solution {
public:
	int coinChange(vector<int>& coins, int amount) {
        vector<int> dp(amount + 1, amount + 1);
        dp[0] = 0;
        for(int i = 1; i <= amount; ++i){
            for(const auto& coin : coins){
                if(i - coin < 0) continue;
                dp[i] = min(dp[i], 1 + dp[i - coin]);
            }
        }
        return dp[amount] == amount + 1 ? -1 : dp[amount];
	}
};
```

#### 518. 零钱兑换 II

下面是一个错误代码：
```cpp
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        vector<int> dp(amount + 1, 0);
        dp[0] = 1;
        for(int i = 1; i <= amount; ++i){
            for(const auto& coin : coins){
                if(i - coin < 0) continue;
                dp[i] += dp[i - coin];
            }
        }
        return dp[amount];
    }
};
```
可以看到，上述代码和题322基本相同，但是得到的答案却是错误的。因为上述代码求得的是所有可能的排列数，而不是应该求的组合数。

这个题其实是典型的完全背包问题，状态转移方程：
```cpp
dp[i][j] = dp[i - 1][j]; // if j < nums[i]
dp[i][j] = dp[i - 1][j] + dp[i][j - coins[i]]; // if j >= nums[i]
```
初始状态：
```cpp
dp[...][0] = 1;
```
AC代码如下：
```cpp
class Solution{
public:
    int change(int amount, vector<int>& coins){
        vector<vector<int>> dp(coins.size() + 1, vector<int>(amount + 1, 0));
        for(int i = 0; i <= coins.size(); ++i) dp[i][0] = 1;
        for(int i = 1; i <= coins.size(); ++i){
            for(int j = 1; j <= amount; ++j){
                if(j < coins[i - 1]) dp[i][j] = dp[i - 1][j];
                else dp[i][j] = dp[i - 1][j] + dp[i][j - coins[i - 1]];
            }
        }
        return dp[coins.size()][amount];
    }
}
```



#### 887. 鸡蛋掉落

动态规划经典面试题

下面的简单的记忆化递归方法：

```cpp
class Solution {
public:
    vector<vector<int>> dp;
    int dfs(int K, int N){
        if(K == 1) return N;    // 如果只有一个鸡蛋，则需要从一层往上线性扫描，所以最坏情况需要 N 次。(鸡蛋没摔坏可以继续用)
        if(N == 0) return 0;    // 如果需测试楼层数为 0，则返回0。
        if(dp[K][N]) return dp[K][N];   
        int res = INT32_MAX;
        for(int i = 1; i <= N; ++i){
            // dfs(K, N - i) 表示剩余鸡蛋数目为K，在第 i 层测试的时候鸡蛋没坏，因此 K = K，N = N - i
            // dfs(K - 1， i - 1) 表示剩余鸡蛋数目为K，在第 i 层测试的时候鸡蛋坏了，因此 K = K - 1， N = i - 1
            res = min(res, max(dfs(K, N - i), dfs(K - 1, i - 1)) + 1);
        }
        dp[K][N] = res;
        return res;        
    }
    int superEggDrop(int K, int N) {
        dp = vector<vector<int>>(K + 1, vector<int>(N + 1, 0));
        return dfs(K, N);
    }
};

```
由于一共有`KN`个状态，因此上述代码的时间复杂度为`O(KN2)`，线上会超时。

通过对上述代码进行二分优化可以将时间复杂度优化到`O(KNlgN)`：
```cpp
class Solution {
public:
    vector<vector<int>> dp;
    int dfs(int K, int N){
        if(K == 1) return N;
        if(N == 0) return 0;
        if(dp[K][N]) return dp[K][N];
        int lo = 1, hi = N;
        while (lo + 1 < hi) {
            int mid = (lo + hi) / 2;
            int t1 = dfs(K - 1, mid - 1);
            int t2 = dfs(K, N - mid);
            if (t1 < t2)
                lo = mid;
            else if (t1 > t2)
                hi = mid;
            else
                lo = hi = mid;
        }
        int res = 1 + min(max(dfs(K-1, lo-1), dfs(K, N-lo)), max(dfs(K-1, hi-1), dfs(K, N-hi)));
        dp[K][N] = res;
        return res;        
    }
    int superEggDrop(int K, int N) {
        dp = vector<vector<int>>(K + 1, vector<int>(N + 1, 0));
        return dfs(K, N);
    }
};
```

#### 416. 分割等和子集

0-1背包问题变形，实际上就是求能否找到若干个元素，使它们的和恰好为总和的一半。

`dp[i][j]`表示对于前`i`个元素，能否找到若干个元素，使得它们的和为`j`。

因此，`dp[N][target]`就是我们要的结果。

状态转移：
```cpp
dp[i][j] = true;                                    // if j == nums[i]
dp[i][j] = dp[i - 1][j];                            // if j < nums[i]
dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i]];  // if j > nums[i]
```
```cpp
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = accumulate(nums.begin(), nums.end(), 0);
        if(sum & 1) return false;
        int target = sum / 2;
        vector<vector<bool>> dp(nums.size(), vector<bool>(target + 1, false));
        if(nums[0] == target) dp[0][nums[0]] = true;    // 第一行在 j == num[0] 时为真
        for(int i = 1; i < nums.size(); ++i){
            for(int j = 1; j <= target; ++j){
                if(j == nums[i]) dp[i][j] = true;
                else if(j < nums[i]) dp[i][j] = dp[i - 1][j];
                else dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i]];
            }
            if(dp[i][target]) return true;
        }
        return dp[nums.size() - 1][target];
    }
};
```

#### 72. 编辑距离

动态规划问题。

定义状态：`dp[i][j]`表示`word1`的前`i`个字符到`word2`的前`j`个字符的距离。

状态转移方程：
```cpp
dp[i][j] = dp[i - 1][j - 1]; // if word1[i] == word2[j]
dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
```

初始状态：
```cpp
dp[i][0] = i;
dp[0][j] = j;
```
其中，`dp[i][0]`为长度为`i`的`word1`转换到空字符串`word2`的距离（`i`次插入）。`dp[0][j]`同理。

AC 代码：
```cpp
class Solution{
public:
    int minDistance(string word1, strign word2){
        vector<vector<int>> dp(word1.size(), vector<int>(word2.size(), 0));
        for(int i = 1; i <= word1.size(); ++i) dp[i][0] = i;
        for(int j = 1; j <= word2.size(); ++j) dp[0][j] = j;
        for(int i = 1; i <= word1.size(); ++i){
            for(int j = 1; j <= word2.size(); ++j){
                if(word1[i - 1] == word2[j - 1]) dp[i][j] = dp[i - 1][j - 1];
                else dp[i][j] = min(min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
            }
        }
        return dp[word1.size()][word2.size()];
    }
}

```

#### 1143.最长公共子序列（LCS）

典型动态规划问题。

定义状态：`dp[i][j]`表示`text1`的前`i`个字符和`text2`的前`j`个字符的最长公共子序列。

状态转移方程：
```cpp
dp[i][j] = dp[i - 1][j - 1] + 1; // if text1[i] == text2[j]
dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]); // if text1[i] != text2[j]
```

初始状态：
```cpp
dp[0][...] = 0;
dp[...][0] = 0;
```

AC 代码：
```cpp
class Solution{
public:
    int longestCommonSubsequence(string text1, string text2){
        vector<vector<int>> dp(text1.size() + 1, vector<int>(text2.size() + 1, 0));
        for(int i = 1; i <= text1.size(); ++i){
            for(int j = 1; j <= text2.size(); ++j){
                if(text1[i - 1] == text2[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1;
                else dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
        return dp[text1.size()][text2.size()];
    }
}
```

#### 516. 最长回文子序列

子序列问题，一般用动态规划进行解决。

定义状态：`dp[i][j]`表示字符串`s`的`i`到`j`位置的最长回文子序列长度。

状态转移：
```cpp
dp[i][j] = dp[i + 1][j - 1] + 2; // if s[i] == s[j]
dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]); // if s[i] == s[j]
```

初始状态：
```cpp
dp[i][i] = 1 for i in range(s.size());
```

AC 代码：
```cpp
class Solution{
public:
    int longestPalindromeSubseq(string s){
        vector<vector<int>> dp(s.size(), vector<int>(s.size(), 0);
        for(int i = 0; i < s.size(); ++i) dp[i][i] = 1;
        for(int i = s.size() - 1; i >= 0; --i){
            for(int j = i + 1; j < s.size(); ++j){
                if(s[i] == s[j]) dp[i][j] = dp[i + 1][j - 1];
                else dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
            }
        }
        return dp[0][s.size() - 1];
    }
};
```
其中，由于`dp[i][j]`与`dp[i + 1][j - 1]`有关，因此`i`的顺序是`n-1 -> 0`，而`j`的顺序是`i -> n-1`。

#### 877. 石子游戏

博弈论问题，可用动态规划解决。

状态定义:
```cpp
dp[i][j][0];    // 在piles[i]...piles[j]中，先手能够获得的最大数量
dp[i][j][1];    // 在piles[i]...piles[j]中，后手能够获得的最大数量
```

状态转换：
```cpp
if(piles[i] + dp[i + 1][j][1] > piles[j] + dp[i][j - 1][1]){
    dp[i][j][0] = piles[i] + dp[i + 1][j][1];
    dp[i][j][1] = dp[i + 1][j][0];
}else{
    dp[i][j][0] = piles[j] + dp[i][j - 1][1];
    dp[i][j][1] = dp[i][j - 1][0];
}
```

初始状态：
```cpp
for i in range(piles.size()):
    dp[i][i][0] = piles[i];
    dp[i][i][1] = 0;
```

AC 代码：
```cpp
class Solutioin{
    int stoneGame(vector<int>& piles){
        int n = piles.size();
        vector<vector<vector<int>>> dp(n, vector<vector<int>>(n, vector<int>(2, 0)));
        for(int i = 0; i < n; ++i){
            dp[i][i][0] = piles[i];
            dp[i][i][1] = 0;
        }
        for(int i = n - 1; i > 0; --i){
            for(int j = i + 1; j < n; ++j){
                int left = piles[i] + dp[i + 1][j][1];
                int right = piles[j] + dp[i][j + 1][1];
                if(left > right){
                    dp[i][j][0] = left;
                    dp[i][j][1] = dp[i + 1][j][0];
                }else{
                    dp[i][j][0] = right;
                    dp[i][j][1] = dp[i][j - 1][0];
                }
            }
        }
        return dp[0][n - 1];
    }
};
```

#### 435. 无重叠区间

贪心算法

```cpp
class Solution{
    int eraseOverlapIntervals(vector<vector<int>>& intervals){
        if(intervals.size() == 0) return 0;
        sort(intervals.begin(), intervals.end(), [](const auto& a, const auto& b){
            return a[1] < b[1];
        });
        int cnt = 1, end = intervals[0][1];
        for(int i = 1; i < intervals.size(); ++i){
            if(intervals[i][0] >= end){
                cnt++; end = intervals[i][1];
            }
        }
        return intervals.size() - cnt;
    }
};
```

#### 739. 每日温度

单调栈

```cpp
class Solution{
public:
    vector<int> dailyTemperatures(vector<int>& T){
        vector<int> ans(T.size());
        stack<int> s;
        for(int i = T.size() - 1; i >= 0; --i){
            while(!s.empty() && T[s.top()] <= T[i]) s.pop();
            ans[i] = s.empty() ? 0 : s.top() - i;
            s.push(i);
        }
        return ans;
    }
};
```

#### 496. 下一个更大的元素

单调栈

```cpp
class Solution {
public:
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        unordered_map<int, int> Next;
        stack<int> s;
        for(int i = nums2.size() - 1; i >= 0; --i){
            while(!s.empty() && s.top() <= nums2[i]) s.pop();
            Next[nums2[i]] = s.empty() ? -1 : s.top();
            s.push(nums2[i]);
        }
        vector<int> ans;
        for(const auto& num : nums1){
            ans.push_back(Next[num]);
        }
        return ans;
    }
};
```

#### 503. 下一个更大的元素II

单调栈

```cpp
class Solution {
public:
    vector<int> nextGreaterElements(vector<int>& nums) {
        int n = nums.size();
        for(int i = 0; i < n; ++i) nums.push_back(nums[i]);
        vector<int> ans(n);
        stack<int> s;
        for(int i = 2 * n - 1; i >= 0; --i){
            while(!s.empty() && s.top() <= nums[i]) s.pop();
            if(i < n) ans[i] = s.empty() ? -1 : s.top();
            s.push(nums[i]);
        }
        return ans;
    }
};
```

#### 78. 子集

回溯法生成子集

```cpp
class Solution {
public:
    vector<vector<int>> ans;
    void bst(vector<int>& nums, int curr, vector<int>& track){
        ans.push_back(track);
        for(int i = curr; i < nums.size(); ++i){
            track.push_back(nums[i]);
            bst(nums, i + 1, track);
            track.pop_back();
        }
    }

    vector<vector<int>> subsets(vector<int>& nums) {
        vector<int> track;
        bst(nums, 0, track);
        return ans;
    }
};
```
#### 77. 组合

回溯法生成组合（组合实际上和子集等价）

```cpp
class Solution {
public:
    vector<vector<int>> ans;
    void combine(int n, int k, int curr, vector<int>& track){
        if(track.size() == k) ans.push_back(track);
        for(int i = curr; i <= n; ++i){
            track.push_back(i);
            combine(n, k, i + 1, track);
            track.pop_back();
        }
    }
    vector<vector<int>> combine(int n, int k) {
        vector<int> track;
        combine(n, k, 1, track);
        return ans;
    }
};
```
#### 46. 全排列

回溯法生成全排列

解法1：
```cpp
class Solution {
public:
    vector<vector<int>> ans;
    void permute(vector<int>& nums, vector<int>& track, vector<bool>& vis){
        if(track.size() == nums.size()) ans.push_back(track);
        for(int i = 0; i < nums.size(); ++i){
            if(vis[i]) continue;
            track.push_back(nums[i]); vis[i] = true;
            permute(nums, track, vis);
            track.pop_back(); vis[i] = false;
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        vector<int> track;
        vector<bool> vis(nums.size(), false);
        permute(nums, track, vis);
        return ans;
    }
};
```

解法2：
```cpp
class Solution {
public:
    vector<vector<int>> ans;
    void permute(vector<int>& nums, int curr){
        if(curr == nums.size()) ans.push_back(nums);
        for(int i = curr; i < nums.size(); ++i){
            swap(nums[curr], nums[i]);
            permute(nums, curr + 1);
            swap(nums[curr], nums[i]);
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        permute(nums, 0);
        return ans;
    }
};
```

#### 76. 最小覆盖子串

典型滑动窗口题目

```cpp
class Solution{
public:
    string minWindow(string s, string t){
        int* window = new int[256]();
        int* target = new int[256]();
        for(const auto& ch : t) target[ch]++;
        int need = count_if(target, target + 256, [](const int& t) { return t > 0; });
        int left = 0, right = 0, start = 0, end = INT32_MAX;
        while(right < s.size()){
            char& ch = s[right++];
            window[ch]++;
            if(window[ch] == target[ch]) need--;
            while(!need){
                if(right - left < end - start){
                    start = left;
                    end = right;
                }
                char& ch = s[left++];
                window[ch]--;
                if(window[ch] < target[ch]) need++;
            }
        }
        return end > s.size() ? "" : s.substr(start, end - start);
    }
};
```

#### 567. 字符串的排列

滑动窗口

```cpp
class Solution{
public:
    bool checkInclusion(string s1, string s2){
        if(s1.size() > s2.size()) return false;
        int* window = new int[256]();
        int* target = new int[256]();
        for(const char& ch : s1) target[ch]++;
        int need = count_if(target, target + 256, [](const int& t){ return t > 0; });
        int left = 0, right = 0;
        while(right < s2.size()){
            char& ch = s2[right++];
            window[ch]++;
            if(window[ch] == target[ch]) need--;
            while(!need){
                if(right - left == s1.size()) return true;
                char& ch = s2[left++];
                window[ch]--;
                if(window[ch] < target[ch]) need++;
            }
        }
        return false;
    }
};
```

#### 438. 找到所有字符串中的字母异位词

滑动窗口

```cpp
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        if(s.size() < p.size()) return {};
        vector<int> ans;
        int* window = new int[256]();
        int* target = new int[256]();
        for(const char& ch : p) target[ch]++;
        int need = count_if(target, target + 256, [](const int& t) { return t > 0; });
        int left = 0, right = 0;
        while(right < s.size()){
            char& ch = s[right++];
            window[ch]++;
            if(window[ch] == target[ch]) need--;
            while(!need){
                if(right - left == p.size()) ans.push_back(left);
                char& ch = s[left++];
                window[ch]--;
                if(window[ch] < target[ch]) need++;
            }
        }
        return ans;
    }
};
```

#### 3. 无重复字符的最长子串

滑动窗口

```cpp
class Solution{
public:
    int lengthOfLongestSubstring(string s){
        int* window = new int[256]();
        int multi = 0;
        int left = 0, right = 0, longest = 0;
        while(right < s.size()){
            char& ch = s[right++];
            window[ch]++;
            if(window[ch] == 2) multi++;
            while(multi){
                char& ch = s[left++];
                window[ch]--;
                if(window[ch] == 1) multi--;
            }
            if(right - left > longest) longest = right - left;
        }
        return longest;
    }
};
```

#### 969. 煎饼排序

```cpp
class Solution {
public:
    vector<int> pancakeSort(vector<int>& A) {
        vector<int> ans;
        int len = A.size();
        while(len > 1){
            int pos = max_element(A.begin(), A.begin() + len) - A.begin();
            if(pos == len - 1){
                len--; continue;
            };
            ans.push_back(pos + 1);
            reverse(A.begin(), A.begin() + pos + 1);
            reverse(A.begin(), A.begin() + len);
            ans.push_back(len--);
        }
        return ans;
    }
};
```

#### 560. 和为k的子数组

前缀和 + twoSum

```cpp
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        vector<int> pre(nums.size() + 1, 0);
        for(int i = 1; i <= nums.size(); ++i) pre[i] = nums[i - 1] + pre[i - 1];
        int ans = 0;
        unordered_map<int, int> map;
        map[0] = 1;
        for(int i = 1; i < pre.size(); ++i){
            if(map.count(pre[i] - k)) ans += map[pre[i] - k];
            map[pre[i]]++;
        }
        return ans;
    }
};
```

#### 56. 合并区间

贪心算法

```cpp
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        if(intervals.size() == 0) return {};
        sort(intervals.begin(), intervals.end(), [](const auto& a, const auto& b){
            return a[0] < b[0];
        });
        vector<vector<int>> ans;
        int start = intervals[0][0], end = intervals[0][1];
        for(int i = 1; i < intervals.size(); ++i){
            if(intervals[i][0] > end){
                ans.push_back({start, end});
                start = intervals[i][0];
            }
            end = max(end, intervals[i][1]);
        }
        ans.push_back({start, end});
        return ans;
    }
};
```

#### 986. 区间列表的交集

```cpp
class Solution {
public:
    vector<vector<int>> intervalIntersection(vector<vector<int>>& A, vector<vector<int>>& B) {
        vector<vector<int>> res;
        int left = 0, right = 0;
        while(left < A.size() && right < B.size()){
            if(A[left][1] >= B[right][0] && B[right][1] >= A[left][0]){
                res.push_back({max(A[left][0], B[right][0]), min(A[left][1], B[right][1])});
            }
            A[left][1] > B[right][1] ? right++ : left++;
        }
        return res;
    }
};
```

#### 42. 接雨水

当前位置`curr`能装的水和下面三者有关：
1. `height[..curr-1]`的最大值`lmax`。
2. `height[curr+1..]`的最大值`rmax`。
3. height[curr]。

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        if(height.size() <= 2) return 0;
        vector<int> rmax(height.size(), 0);
        for(int i = height.size() - 2; i >= 0; --i) rmax[i] = max(rmax[i + 1], height[i + 1]);
        int lmax = 0, res = 0;
        for(int i = 0; i < height.size(); ++i){
            int minmax = min(lmax, rmax[i]);
            if(minmax > height[i]) res += curr - height[i];
            lmax = max(lmax, height[i]);
        } 
        return res;
    }
};
```

#### 55. 跳跃游戏

第一想法是动态规划，可惜会超时。

第二想法是借鉴区间合并的思想，直接贪心即可。

```cpp
// class Solution {
// public:
//     bool canJump(vector<int>& nums) {
//         vector<bool> dp(nums.size(), false);
//         dp[nums.size() - 1] = true;
//         for(int i = nums.size() - 2; i >= 0; --i){
//             for(int j = 1; j <= nums[i] && i + j < nums.size(); ++j){
//                 if(dp[i + j] == true){
//                     dp[i] = true; break;
//                 }
//             }
//         }
//         return dp[0];
//     }
// };
class Solution{
public:
    bool canJump(vector<int>& nums){
        int pos = nums[0];
        for(int i = 1; i < nums.size(); ++i){
            if(i > pos) return false;
            pos = max(pos, i + nums[i]);
        }
        return true;
    }
};
```

#### 45. 跳跃游戏II

贪心选择：每次选择能跳到的位置中，能跳距离最远的值。

```cpp
class Solution {
public:
    int jump(vector<int>& nums) {
        int step = 0, pos = 0, farthest = 0;
        for(int i = 0; i < nums.size(); ++i){
            farthest = max(farthest, nums[i] + i);
            if(i == pos && i != nums.size() - 1){
                pos = farthest; step++;
            }
        }
        return step;
    }
};
```

上面用了`farthest`来维护最远位置，当`i == pos`时，更新下一次跳的位置`pos`。


#### 84. 柱状图中最大的矩形

双向单调栈

```cpp
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        stack<int> s, s1;
        heights.insert(heights.begin(), 0);
        heights.push_back(0);
        vector<int> left(heights.size(), 0), right(heights.size(), 0);
        for(int i = heights.size() - 1; i >= 0; --i){
            while(!s.empty() && heights[s.top()] >= heights[i]) s.pop();
            if(!s.empty()) right[i] = s.top() - i - 1;
            s.push(i);
        }
        for(int i = 0; i < heights.size(); ++i){
            while(!s1.empty() && heights[s1.top()] >= heights[i]) s1.pop();
            if(!s1.empty()) left[i] = i - s1.top() - 1;
            s1.push(i);
        }
        int ans = 0;
        for(int i = 1; i < heights.size() - 1; ++i){
            ans = max(ans, heights[i] * (left[i] + right[i] + 1));
        }
        return ans;
    }
};
```

#### 239. 滑动窗口最大值

类似单调栈，这里是单调双端队列

```cpp
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        deque<int> Q;
        vector<int> ans;
        for(int i = 0; i < nums.size(); ++i){
            if(!Q.empty() && i - Q.front() >= k) Q.pop_front();
            while(!Q.empty() && nums[i] >= nums[Q.back()]) Q.pop_back();
            Q.push_back(i);
            if(i >= k - 1) ans.push_back(nums[Q.front()]);
        }
        return ans;
    }
};
```

#### 312. 戳气球
```cpp
class Solution {
public:
    int maxCoins(vector<int>& nums) {
        int n = nums.size();
        // 添加两侧的虚拟气球
        vector<int> points(n + 2);
        points[0] = points[n + 1] = 1;
        for (int i = 1; i <= n; i++) {
            points[i] = nums[i - 1];
        }
        // base case 已经都被初始化为 0
        vector<vector<int>> dp(n + 2, vector<int>(n + 2, 0));
        // 开始状态转移
        // i 应该从下往上
        for (int i = n; i >= 0; i--) {
            // j 应该从左往右
            for (int j = i + 1; j < n + 2; j++) {
                // 最后戳破的气球是哪个？
                for (int k = i + 1; k < j; k++) {
                    // 择优做选择
                    dp[i][j] = max(
                        dp[i][j], 
                        dp[i][k] + dp[k][j] + points[i] * points[j] * points[k]
                    );
                }
            }
        }
        return dp[0][n + 1];
    }
};
```

#### 301. 删除无效的括号

```cpp
class Solution {
public:
    vector<string> removeInvalidParentheses(string s) {
        vector<string> ans;
        unordered_map<string, int> vis;
        vis[s] = 1;
        queue<string> Q;
        Q.push(s);
        bool found = false;
        while (!Q.empty()) {
            string curr = Q.front(); Q.pop();
            if (isValid(curr)){
                ans.push_back(curr);
                found = true;
            }
            if (found) continue;
            for (int i = 0; i < curr.size(); ++i) {
                if (curr[i] != '(' && curr[i] != ')') continue;
                string str = curr.substr(0, i) + curr.substr(i + 1);
                if(!vis.count(str)) {
                    Q.push(str);
                    vis[str] = 1;
                }
            }
        }
        return ans;
    }
    bool isValid(string t) {
        int cnt = 0;
        for (int i = 0; i < t.size(); ++i) {
            if (t[i] == '(') ++cnt;
            else if (t[i] == ')' && --cnt < 0) 
                return false;
        }
        return cnt == 0;
    }
};
```

#### 85. 最大矩形

复用 84 题（柱状图中的最大矩形）的算法

```cpp
class Solution {
public:
    int largestRectangleArea(vector<int> heights) {
        stack<int> s, s1;
        heights.insert(heights.begin(), 0);
        heights.push_back(0);
        vector<int> left(heights.size(), 0), right(heights.size(), 0);
        for(int i = heights.size() - 1; i >= 0; --i){
            while(!s.empty() && heights[s.top()] >= heights[i]) s.pop();
            if(!s.empty()) right[i] = s.top() - i - 1;
            s.push(i);
        }
        for(int i = 0; i < heights.size(); ++i){
            while(!s1.empty() && heights[s1.top()] >= heights[i]) s1.pop();
            if(!s1.empty()) left[i] = i - s1.top() - 1;
            s1.push(i);
        }
        int ans = 0;
        for(int i = 1; i < heights.size() - 1; ++i){
            ans = max(ans, heights[i] * (left[i] + right[i] + 1));
        }
        return ans;
    }
    int maximalRectangle(vector<vector<char>>& matrix) {
        if(matrix.size() == 0) return 0;
        vector<vector<int>> height(matrix.size(), vector<int>(matrix[0].size()));
        int ans = 0;
        for(int i = 0; i < matrix.size(); ++i){
            for(int j = 0; j < matrix[i].size(); ++j){
                if(i == 0) height[i][j] = matrix[i][j] - '0';
                else if(matrix[i][j] == '0') height[i][j] = 0;
                else height[i][j] = height[i - 1][j] + 1;
            }
            ans = max(ans, largestRectangleArea(height[i]));
        }
        return ans;
    }
};
```