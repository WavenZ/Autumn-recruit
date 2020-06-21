#### 1. 两数之和 (leetcode-01)

题目：https://leetcode-cn.com/problems/two-sum/


思路：遍历的时候将已经访问过的元素存进哈希表`exist`，对于当前元素`numbers[i]`，只需要判断`target - numbers[i]`是否存在即可。

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        unordered_map<int, int> exist;
        for(int i = 0; i < numbers.size(); ++i){
            if(exist.count(target - numbers[i])){
                return {exist[target - numbers[i]], i};
            }
            exist[numbers[i]] = i;
        }
        return {};
    }
};
```

#### 2. 两数相加 (leetcode-02)

题目：https://leetcode-cn.com/problems/add-two-numbers/

思路：递归和非递归都容易实现，需要注意的是进位的处理，最终长度可能比两个链表都长。

```cpp
class Solution {
public:
    ListNode* func(ListNode* l1, ListNode* l2, int c){
        if(l1 == nullptr) return c ? func(l2, new ListNode(c), 0) : l2;
        if(l2 == nullptr) return c ? func(l1, new ListNode(c), 0) : l1;
        int sum = l1->val + l2->val + c;
        l1->val = sum % 10;
        l1->next = func(l1->next, l2->next, sum / 10);
        return l1;
    }
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        return func(l1, l2, 0);
    }
};
```

#### 3. 无重复字符的最长字串 (leetcode-03)

题目：https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/

思路：典型的滑动窗口题目，直接默写就行了。

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

#### 4. 寻找两个正序数组的中位数 (leetcode-04)

题目：https://leetcode-cn.com/problems/median-of-two-sorted-arrays/


#### 5. 最长回文子串 (leetcode-05)

题目：https://leetcode-cn.com/problems/longest-palindromic-substring/

思路：动态规划、中心扩散、Manacher算法（非常规）

```cpp
// 1. 动态规划
class Solution {
public:
    string longestPalindrome(string s) {
        if(s.size() == 0) return "";
        vector<vector<bool>> dp(s.size(), vector<bool>(s.size(), true));
        int start = 0, len = 1;
        for (int l = 1; l < s.size(); l++) {
            for (int i = 0; i + l < s.size(); i++) {
                int j = i + l;
                if (dp[i + 1][j - 1] && s[i] == s[j]) {
                    start = i;
                    len = l + 1;
                }else dp[i][j] = false;
            }
        }
        return s.substr(start, len);
    }
};
```
```cpp
// 2. 中心扩散
class Solution {
public:
    string longestPalindrome(string s) {
        if(s.size() == 0) return "";
        int start = 0, maxLen = 1;
        int left, right;
        for(int i = 0; i < s.size(); ++i){
            for(int j = 0; j < 2; ++j){ // 0->odd 1->even
                left = i; right = i + j;
                while(left >= 0 && right < s.size() && s[left] == s[right]){
                    left--; right++;
                }
                if(right - left - 1 > maxLen){
                    maxLen = right - left - 1; start = left + 1;
                }
            }
        }
        return s.substr(start, maxLen);
    }
};
```

线上中心扩散法比动态规划快很多。

#### 6. 正则表达式匹配 (hard) (leetcode-10)

题目：https://leetcode-cn.com/problems/regular-expression-matching/

思路：动态规划题目，难点在于状态转换方程。

```cpp
class Solution {
public:
    bool isMatch(string s, string p) {
        s = " " + s; p = " " + p;
        int m = s.size(), n = p.size();
        vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
        dp[0][0] = true;
        for(int i = 1; i <= m; ++i){
            for(int j = 1; j <= n; ++j){
                if(s[i - 1] == p[j - 1] || p[j - 1] == '.'){
                    dp[i][j] = dp[i - 1][j - 1];
                }
                else if(p[j - 1] == '*'){
                    if(s[i - 1] != p[j - 2] && p[j - 2] != '.')
                        dp[i][j] = dp[i][j - 2];
                    else{
                        dp[i][j] = dp[i][j - 1] || dp[i][j - 2] || dp[i - 1][j];
                    }
                }
            }
        }
        return dp[m][n];
    }
};
```

#### 7. 盛最多水的容器 (leetcode-11)

题目：https://leetcode-cn.com/problems/container-with-most-water/

注意：这个题和接雨水（leetcode42）、柱状图中的最大矩形（leetcode84）类似，可对比解法的异同。

思路：最简单的方法是暴力枚举起点和终点，显然会超时。如果使用双指针法，那么指针如何移动便是本题的关键。可以（不容易）观察到，移动`height`较高的指针只能使得容量减小，而移动`height`较高的指针才可能使得容量增加，因此需要每次移动`height`较小的指针。

```cpp
class Solution {
public:
    int maxArea(vector<int>& height) {
        int ans = 0;
        int left = 0, right = height.size() - 1;
        while(left < right){
            ans = max(ans, (right - left) * min(height[left], height[right]));
            height[left] < height[right] ? left++ : right--;
        }
        return ans;
    }
};
```

#### 8. 三数之和 (leetcode-15)

题目：https://leetcode-cn.com/problems/3sum/

思路：暴力法的时间复杂度是`O(n3)`，显然会超时。可以考虑使用双指针法，固定第一个指针，第二、三个指针进行移动。此时，需要先将数组排序以便于指针移动。

```cpp
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        if(nums.size() < 3) return {};
        sort(nums.begin(), nums.end());
        vector<vector<int>> ans;
        for(int i = 0; i < nums.size();){
            int left = i + 1, right = nums.size() - 1;
            while(left < right){
                int sum = nums[i] + nums[left] + nums[right];
                if(sum == 0){
                    ans.push_back({nums[i], nums[left], nums[right]});
                    while(++left < right && nums[left] == nums[left - 1]);
                    while(left < --right && nums[right] == nums[right + 1]);
                } 
                else if(sum < 0){
                    while(++left < right && nums[left] == nums[left - 1]);
                }else{
                    while(left < --right && nums[right] == nums[right + 1]);
                }
            }
            while(++i < nums.size() && nums[i] == nums[i - 1]);
        }
        return ans;
    }
};
```

#### 9. 电话号码的字母组合 (leetcode-17)

题目：https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/

思路：典型的回溯法题目，直接默写即可。

```cpp
class Solution {
public:
    vector<string> ans;
    void backtrack(vector<vector<char>>& vec, string& digits, int curr, string& track){
        if(curr == digits.size()){
            ans.push_back(track);
            return;
        } 
        int digit = digits[curr] - '0';
        for(int i = 0; i < vec[digit].size(); ++i){
            char& ch = vec[digit][i];
            track.push_back(ch);
            backtrack(vec, digits, curr + 1, track);
            track.pop_back();
        }
    }
    vector<string> letterCombinations(string digits) {
        if(digits.size() == 0) return {};
        vector<vector<char>> vec = {{}, {}, {'a', 'b', 'c'}, {'d', 'e', 'f'}, 
        {'g', 'h', 'i'}, {'j', 'k', 'l'}, {'m', 'n', 'o'}, {'p', 'q', 'r', 's'}, 
        {'t', 'u', 'v'}, {'w', 'x', 'y', 'z'}};
        string track;
        backtrack(vec, digits, 0, track);
        return ans;
    }
};
```

#### 10. 删除链表的倒数第N个节点 (leetcode-19)

题目：https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/

思路：看到链表的倒数节点就应该想到前后指针法，另外还需要注意被删除的节点是头结点的情况。

```cpp
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* fast = head, *slow = head;
        for(int i = 0; i < n; ++i) fast = fast->next;
        if(fast == nullptr) return head->next;
        fast = fast->next;
        while(fast){
            fast = fast->next;
            slow = slow->next;
        }
        ListNode* temp = slow->next;
        slow->next = slow->next->next;
        delete temp;
        return head;

    }
};
```

#### 11. 有效的括号 (leetcode-20)

题目：https://leetcode-cn.com/problems/valid-parentheses/

思路：判断括号的有效性一般用栈。

```cpp
class Solution {
public:
    bool isValid(string s) {
        stack<char> S;
        for(const auto& ch : s){
            switch(ch){
                case ')':{
                    if(!S.empty() && S.top() == '(') S.pop();
                    else return false;
                }break;
                case ']':{
                    if(!S.empty() && S.top() == '[') S.pop();
                    else return false;
                }break;
                case '}':{
                    if(!S.empty() && S.top() == '{') S.pop();
                    else return false;
                }break;
                default: S.push(ch); break;
            }
        }
        return S.empty();
    }
};
```
注意：最后这个`return S.empty();`很重要。

#### 12. 合并两个有序链表 (leetcode-21)

题目：https://leetcode-cn.com/problems/merge-two-sorted-lists/

思路：比较简洁的方法是直接用递归。

```cpp
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if(l1 == nullptr) return l2;
        if(l2 == nullptr) return l1;
        if(l1->val <= l2->val){
            l1->next = mergeTwoLists(l1->next, l2);
            return l1;
        }else{
            l2->next = mergeTwoLists(l1, l2->next);
            return l2;
        }
        return nullptr;
    }
};
```

#### 13. 括号生成 (leetcode-22)

题目：https://leetcode-cn.com/problems/generate-parentheses/

思路：回溯法。对于任何一个中间态，有效的括号须满足：
1. 左括号`"("`的数量大于等于右括号`")"`的数量。
2. 左括号`"("`和右括号`")"`的数量均小于等于`n`。

```cpp
class Solution{
public:
    vector<string> ans;
    void backtrack(string& s, int l, int r){
        if(l == 0 && r == 0){
            ans.push_back(s);
        }
        if(l > 0){
            s.push_back('(');
            backtrack(s, l - 1, r);
            s.pop_back();
        }
        if(r > l){
            s.push_back(')');
            backtrack(s, l, r - 1);
            s.pop_back();
        }
    }
    vector<string> generateParenthesis(int n){
        string s;
        backtrack(s, n, n);
        return ans;
    }
};
```

#### 14. 合并K个排序链表 (leetcode-23)

题目：https://leetcode-cn.com/problems/merge-k-sorted-lists/

思路：前面有一个合并2个排序链表的题目，它的递归解法非常的简洁。对于`K`个链表的合并，需要快速找到`k`个链表的当前节点的最小值，一个可行的方法是使用优先队列。

```cpp
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        using Node = pair<int, ListNode*>;
        priority_queue<Node, vector<Node>, greater<Node>> Q;
        for(const auto& list : lists){
            if(list) Q.push({list->val, list});
        }
        ListNode* head = new ListNode(0);
        ListNode* curr = head;
        while(!Q.empty()){
            ListNode* p = Q.top().second; Q.pop();
            curr->next = p;
            curr = curr->next;
            if(p->next){
                Q.push({p->next->val, p->next});
            }
        }
        return head->next;
    }
};
```
注意：直接`priority_queue<ListNode*> Q`然后再重载`ListNode*`的`<`操作符是不行的。

此外，我们还可以采用分治策略从而复用合并2个有序链表的代码。

#### 15. 下一个排列 (leetcode-31)

题目：https://leetcode-cn.com/problems/next-permutation/

思路：没办法，只能记住。

```cpp
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        if(nums.size() < 2) return;
        int i = nums.size() - 2;
        for(; i >= 0; --i) if(nums[i] < nums[i + 1]) break;
        if(i == -1){
            reverse(nums.begin(), nums.end()); return;
        }
        int j = nums.size() - 1;
        for(; j > i; --j) if(nums[i] < nums[j]) break;
        swap(nums[i], nums[j]);
        reverse(nums.begin() + i + 1, nums.end());
    }
};
```

#### 16. 最长有效括号 (leetcode-32)

题目：https://leetcode-cn.com/problems/longest-valid-parentheses/

思路：leecode官方题解有几种解法，比较简单直观的也就是动态规划了。

1. s[i] = ‘)’且s[i - 1] = ‘(’，也就是字符串形如"……()"，我们可以推出：dp[i] = dp[i − 2] + 2
我们可以进行这样的转移，是因为结束部分的"()"是一个有效子字符串，并且将之前有效子字符串的长度增加了2。

2. s[i] = ‘)’且s[i - 1] = ‘)’，也就是字符串形如".......))"，我们可以推出：
如果s[i - dp[i - 1] - 1] = ‘(’，那么dp[i] = dp[i - 1] + dp[i - dp[i - 1] - 2] + 2

```cpp
class Solution {
public:
    int longestValidParentheses(string s) {
        int ans = 0;
        vector<int> dp(s.size());
        for (int i = 1; i < s.length(); ++i) {
            if (s[i] == ')') {
                if (s[i - 1] == '(') {
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                } else if (i - dp[i - 1] > 0 && s[i - dp[i - 1] - 1] == '(') {
                    dp[i] = dp[i - 1] + ((i - dp[i - 1] - 2) >= 0 ? dp[i - dp[i - 1] - 2] : 0) + 2;
                }
                ans = max(ans, dp[i]);
            }
        }
        return ans;
    }
};
```

#### 17. 搜索旋转排序数组 (leetcode-33)

题目：https://leetcode-cn.com/problems/search-in-rotated-sorted-array/

思路：题目要求时间复杂度是`O(logn)`级别，因此多半是考察二分算法。由于数组是旋转排序的，因此左右边界的移动判断需要考虑`mid`和`target`是否在前后两个部分的同一个部分。

```cpp
class Solution {
public:
    int search(vector<int>& nums, int target) {
        if(nums.size() == 0) return -1;
        int start = nums.front(), end = nums.back();
        int low = 0, high = nums.size() - 1;
        while(low <= high){
            int mid = low + ((high - low) >> 1);
            if(nums[mid] == target) return mid;
            else if(nums[mid] < target){
                if((nums[mid] > end && target > end) || (nums[mid] <= end && target <= end)) low = mid + 1;
                else high = mid - 1;
            }else{
                if((nums[mid] > end && target > end) || (nums[mid] <= end && target <= end)) high = mid - 1;
                else low = mid + 1;
            }
        }
        return -1;
    }
};
```
其中，`(nums[mid] > end && target > end) || (nums[mid] <= end && target <= end)`表示`mid`和`target`在同一个部分。

#### 18. 在排序数组中查找元素的第一个和最后一个位置 (leetcode-34)

题目：https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/

思路：二分查找，`lower_bound`和`upper_bound`的合体，可以直接调用`stl`中的函数，下面是手动实现：

```cpp
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        vector<int> ans;
        int low = 0, high = nums.size() - 1;
        while(low < high){
            int mid = low + ((high - low) >> 1);
            if(nums[mid] < target) low = mid + 1;
            else high = mid;
        }
        if(high < 0 || high >= nums.size() || nums[high] != target) return {-1, -1};
        ans.push_back(high);
        high = nums.size() - 1;
        while(low < high){
            int mid = (low + high + 1) >> 1;
            if(nums[mid] <= target) low = mid;
            else high = mid - 1;
        }
        ans.push_back(low);
        return ans;
    }
};
```
注意：二分查找思路很简单，但是细节很重要。

#### 19. 组合总和 (leetcode-39)

题目：https://leetcode-cn.com/problems/combination-sum/

思路：如果不允许多次选取同一个元素，那么直接默写组合/子集代码即可。这里由于可以多次选取同一个元素，需要对代码进行微调：

```cpp
class Solution {
public:
    vector<vector<int>> ans;
    void backtrack(vector<int>& vec, int curr, vector<int>& track, int target){
        int sum = accumulate(track.begin(), track.end(), 0);
        if(sum >= target){
            if(sum == target)
                ans.push_back(track);
            return;
        } 
        for(int i = curr; i < vec.size(); ++i){
            track.push_back(vec[i]);
            backtrack(vec, i, track, target); // 注意这里是`i`而不是自己生成的`i + 1`，因为可以多次选取统一个元素
            track.pop_back();
        }
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<int> track;
        backtrack(candidates, 0, track, target);
        return ans;
    }
};
```

#### 20. 接雨水 (leetcode-42)

题目：https://leetcode-cn.com/problems/trapping-rain-water/

思路：这个题需要从局部入手：对于第`i`个位置，其能容纳的雨水的大小与其左边的最大值和右边的最大值中较小者有关。因此如果提前统计出`lmax[i]`和`rmax[i]`，则再通过一次遍历即可得到答案。

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
            if(minmax > height[i]) res += minmax - height[i];
            lmax = max(lmax, height[i]);
        } 
        return res;
    }
};
```
注意：`lmax`可以在正向遍历的时候动态更新。

#### 21. 全排列 (leetcode-46)

题目：https://leetcode-cn.com/problems/permutations/

思路：需要做到能够快速默写。

解法1：
```cpp
class Solution {
public:
    vector<vector<int>> ans;
    void permute(vector<int>& nums, int curr){
        if(curr == nums.size()) ans.push_back(nums);
        for(int i = curr; i < nums.size(); ++i){
            swap(nums[i], nums[curr]);
            permute(nums, curr + 1);
            swap(nums[i], nums[curr]);
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        permute(nums, 0);
        return ans;
    }
};
```
解法2：
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
解法3：
```cpp
// 解法 3 就是迭代调用 "15.下一个排列" 的代码。
```


#### 22. 旋转图像 (leetcode-48)

题目：https://leetcode-cn.com/problems/rotate-image/

思路：将图像水平镜像，再转置一下就等价于旋转了。

```cpp
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        for(int i = 0; i < matrix.size() / 2; ++i){
             for(int j = 0; j < matrix.size(); ++j){
                swap(matrix[i][j], matrix[matrix.size() - 1 - i][j]);
            }
        }
        for(int i = 0; i < matrix.size(); ++i){
            for(int j = i + 1; j < matrix[i].size(); ++j){
                swap(matrix[i][j], matrix[j][i]);
            }
        }
    }
};
```

#### 23. 字母异位词分组 (leetcode-49)

题目：https://leetcode-cn.com/problems/group-anagrams/

思路：单词字母排序 + 哈希表

```cpp
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        vector<vector<string>> ans;
        unordered_map<string, int> map;
        for(const auto& str :strs){
            auto temp = str;
            sort(temp.begin(), temp.end());
            if(map.count(temp)){
                ans[map[temp]].push_back(str);
            }else{
                map[temp] = ans.size();
                ans.push_back({});
                ans.back().push_back(str);
            }
        }
        return ans;   
    }
};
```

#### 24. 最大子序和 (leetcode-53)

题目：https://leetcode-cn.com/problems/maximum-subarray/

思路：动态规划或者分治法

1. 分治法 `O(nlogn)`
```cpp
class Solution {
public:
    int calcMid(vector<int>& nums, int start, int end){
        int mid = start + ((end - start) >> 1);
        int lmax = INT32_MIN, rmax = INT32_MIN;
        int lsum = 0, rsum = 0;
        for(int i = mid; i >= start; --i){
            lsum += nums[i];
            lmax = max(lmax, lsum);
        }
        for(int i = mid + 1; i <= end; ++i){
            rsum += nums[i];
            rmax = max(rmax, rsum);
        }
        return lmax + rmax;
    }
    int maxSubArray(vector<int>& nums, int start, int end){
        if(start == end) return nums[start];
        int mid = start + ((end - start) >> 1);
        int lval = maxSubArray(nums, start, mid);
        int rval = maxSubArray(nums, mid + 1, end);
        int mval = calcMid(nums, start, end);
        return max(mval, max(lval, rval));
    }
    int maxSubArray(vector<int>& nums) {
        if(nums.size() == 0) return 0;
        return maxSubArray(nums, 0, nums.size() - 1);
    }
};
```

2. 动态规划 `O(n)`

状态定义：`dp[i]`表示以第`i`个元素结尾的子序列之和的最大值。

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        if(nums.size() == 0) return 0;
        int ans = nums[0];
        int dp = nums[0];
        for(int i = 1; i < nums.size(); ++i){
            dp = max(dp + nums[i], nums[i]);
            ans = max(ans, dp);
        }
        return ans;
    }
};
```

#### 25. 跳跃游戏 (leetcode-55)

题目：https://leetcode-cn.com/problems/jump-game/

思路：这个题用动态规划会超时，更佳的解法是贪心算法。首先考虑从`i`个位置能否跳到最后一个位置，它的解与第`i+1..i+nums[i]`中的`nums[i]`个元素有关。如果使用动态规划进行求解，状态转移方程为：
```cpp
dp[i] = dp[i + 1] || dp[i + 2] ... || dp[i + nums[i]];
```
可以观察到，对于第`i`个位置可以跳的下一个位置`j`，应该选择`j + nums[j]`取到最大的一个位置，因为这个位置是最有可能跳跃到最后的，这就是本题的**贪心选择性质**。

代码实现如下：
```cpp
bool canJump(vector<int>& nums){
    for(int i = 0; i < nums.size() - 1;){
        if(nums[i] == 0) return false;
        int farest = 0, next = i;
        for(int j = 1; j <= nums[i] && i + j < nums.size(); ++j){
            if(i + j + nums[i + j] > farest){
                farest = i + j + nums[i + j];
                next = i + j;
            }
        }
        i = next;
    }
    return true;
}
```
还有一个优化点是`i`到`i + nums[i]`中的后半段会被重复计算，也就是`next`到`i + nums[i]`这一段。经过优化之后可以得到官方题解的贪心解法，也就是在全局维护一个`farest`用以记录位置`i`之前能到的最远距离。

完整代码如下：
```cpp
class Solution{
public:
    bool canJump(vector<int>& nums){
        int farest = nums[0];
        for(int i = 1; i < nums.size(); ++i){
            if(i > farest) return false;
            farest = max(farest, i + nums[i]);
        }
        return true;
    }
};
```
上述代码在遍历`nums`的时候维护一个`farest`表示目前能否到达的最远距离。如果`i`在`++`之后超过了`farest`，表示超过了能到的最远距离还不能到达终点，因此返回`false`。

#### 26. 合并区间 (leetcode-56)

题目：https://leetcode-cn.com/problems/merge-intervals/

思路：这个题和上一道题 “25.跳跃游戏” 有异曲同工之妙，该题几乎可以等价于“能否把所有子区间合并为一个区间”。因此，本题和上一题使用的方法基本类似，可以同贪心算法解答。

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
这里的`intervals[0] > end`和上一题的`i > farest`表示的是同一个意思。

#### 27. 不同路径 (leetcode-62)

题目：https://leetcode-cn.com/problems/unique-paths/

思路：典型的动态规划题目。

```cpp
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> dp(m, vector<int>(n, 1));
        for(int i = m - 2; i >= 0; --i){
            for(int j = n - 2; j >=  0; --j){
                dp[i][j] = dp[i + 1][j] + dp[i][j + 1];
            }
        }
        return dp[0][0];
    }
};
```

#### 28. 最小路径和 (leetcode-64)

题目：https://leetcode-cn.com/problems/minimum-path-sum/submissions/

思路：和上一题差不多，直观的动态规划问题。

```cpp
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<int>> dp(m, vector<int>(n));
        dp[m - 1][n - 1] = grid[m - 1][n - 1];
        for(int i = n - 2; i >= 0; --i) dp[m - 1][i] = grid[m - 1][i] + dp[m - 1][i + 1];
        for(int i = m - 2; i >= 0; --i) dp[i][n - 1] = grid[i][n - 1] + dp[i + 1][n - 1];
        for(int i = m - 2; i >= 0; --i){
            for(int j = n - 2; j >= 0; --j){
                dp[i][j] = grid[i][j] + min(dp[i][j + 1], dp[i + 1][j]);   
            }
        }
        return dp[0][0];
    }
};
```

#### 29. 爬楼梯 (leetcode-70)

题目：https://leetcode-cn.com/problems/climbing-stairs/

思路：动态规划，实际上就是斐波那契数列的解。

```cpp
class Solution {
public:
    int climbStairs(int n) {
        int dp = 1, dp1 = 1, temp;
        for(int i = 2; i <= n; ++i){
            temp = dp1; dp1 += dp; dp = temp;
        }
        return dp1;
    }
};
```

#### 30. 编辑距离 (leetcode-72)

题目：https://leetcode-cn.com/problems/edit-distance/

思路：动态规划。

状态定义：`dp[i][j]`表示`word1`的前`i`个字符转换为`word2`的前`j`个字符的最少操作数。

状态转换：
```cpp
dp[i][j] = dp[i - 1][j - 1] if word1[i] == word2[j]
dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) if word1[i] != word2[j]
// dp[i - 1][j - 1]表示替换，dp[i - 1][j]表示删除，dp[i][j - 1]表示插入
```

完整代码：
```cpp
class Solution {
public:
    int minDistance(string word1, string word2) {
        vector<vector<int>> dp(word1.size() + 1, vector<int>(word2.size() + 1, 0));
        for(int i = 1; i <= word1.size(); ++i) dp[i][0] = i;
        for(int j = 1; j <= word2.size(); ++j) dp[0][j] = j;
        for(int i = 1; i <= word1.size(); ++i){
            for(int j = 1; j <= word2.size(); ++j){
                if(word1[i - 1] == word2[j - 1]) dp[i][j] = dp[i - 1][j - 1];
                else dp[i][j] = 1 + min(min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);
            }
        }
        return dp[word1.size()][word2.size()];
    }
};
```

#### 31. 颜色分类 (leetcode-75)

题目：https://leetcode-cn.com/problems/sort-colors/submissions/

思路：三指针法，其中两个指针分别用来指向`0`的右边界和`1`的左边界，另一个指针用来指向当前元素。

```cpp
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int left = 0, right = nums.size() - 1;
        for(int i = 0; i <= right;){
            if(nums[i] == 1) ++i;
            else if(nums[i] == 0){
                swap(nums[i++], nums[left++]);  // *
            }else swap(nums[i], nums[right--]); // * 注意区别
        }
    }
};
```
其中，`0`和`2`的处理实际上是不一样的，因为`i`之前的值已经处理过，而`i`之后的值没有处理过，`swap(nums[i], nums[right--]`之后还需要处理一次`nums[i]`。

#### 32. 最小覆盖子串 (leetcode-76)

题目：https://leetcode-cn.com/problems/minimum-window-substring/

思路：典型的滑动窗口题目，最好能背下来。

```cpp
class Solution{
public:
    string minWindow(string s, string t){
        vector<int> window(256, 0), target(256, 0);
        for(const auto& ch : t) target[ch]++;
        int need = 256 - count(target.begin(), target.end(), 0);
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

#### 33. 子集 (leetcode-78)

题目：https://leetcode-cn.com/problems/subsets/

思路：子集生成，经典回溯，直接背就完事了。

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

#### 34. 单词搜索 (leetcode-84)

题目：https://leetcode-cn.com/problems/word-search/

思路：回溯法

```cpp
class Solution {
public:
    bool dfs(vector<vector<char>>& board, int r, int c, int curr, string& word){
        if(curr == word.size()) return true;
        if(r < 0 || c < 0 || r >= board.size() || c >= board[0].size() || board[r][c] != word[curr) return false;
        board[r][c] = ' ';
        bool ret = dfs(board, r - 1, c, curr + 1, word) ||
                   dfs(board, r, c + 1, curr + 1, word) ||
                   dfs(board, r + 1, c, curr + 1, word) ||
                   dfs(board, r, c - 1, curr + 1, word);
        board[r][c] = word[curr];
        return ret;
    }
    bool exist(vector<vector<char>>& board, string word) {
        for(int r = 0; r < board.size(); ++r){
            for(int c = 0; c < board[0].size(); ++c){
                if(dfs(board, r, c, 0, word)) return true;
            }
        }
        return false;
    }
};
```

#### 35. 柱状图中的最大矩形 (leetcode-85)

题目：https://leetcode-cn.com/problems/largest-rectangle-in-histogram/

思路：这个题和“20.接雨水”题目很类似，同样需要从局部入手。对于第`i`个柱子，完全包含该柱子的最大矩形与左边和右边最后一个不小于`heights[i]`的位置有关。因此需要找到每个柱子左边和右边高度下降的位置，这里可以借助单调栈。（单调栈最直观的例子是“100.每日温度”题目）

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
            // cout << ans << " " << left[i] << " " << right[i] << endl;;
        }
        return ans;
    }
};
```

#### 36. 最大矩形 (leetcode-94)

题目：https://leetcode-cn.com/problems/maximal-rectangle/

思路：初看此题可能会觉得这个题用动态规划很容易解决，但实际上这个题的动态规划解法的状态转换很难推导出来。换个思路可以发现这个题和上一个题“35. 柱状图中的最大矩形”实际上是非常类似的：对于第`0-i`行，求最大矩形实际上等价于求35题所说的柱状图的最大矩形。

例如对于：
```cpp
[
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
```
求其前三行的最大矩形时，可以把前三行看作柱状图：
```cpp
  ["1"," ","1"," "," "]
  ["1"," ","1","1","1"]
  ["1","1","1","1","1"]
```
因此，本题可以服用35题的代码进行求解。

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

#### 37. 二叉树的中序遍历 (leetcode-94)

题目：https://leetcode-cn.com/problems/binary-tree-inorder-traversal/

思路：这个题要求使用迭代的方法实现，通常递归改迭代都是利用栈实现的：

```cpp
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> S;
        TreeNode* node = root;
        while(node != nullptr || !S.empty()){
            while(node != nullptr){
                S.push(node);
                node = node->left;
            }
            node = S.top(); S.pop();
            res.push_back(node->val);
            node = node->right;
        }
        return res;
    }
};
```
注意：前序遍历和后序遍历的迭代版本都比中序遍历难。

#### 38. 不同的二叉搜索树 (leetcode-96)

题目：https://leetcode-cn.com/problems/unique-binary-search-trees/submissions/

思路：首先确定根节点，有`n`种可能。假如根节点为`i`，则可能的二叉搜索树数量`num[i] = numTree(i - 1) * numTree(n - i)`，因此总的二叉搜索树数目为`num[i]`求和。

为了减少重复的计算，可以采用自底向上的动态规划求解：

```cpp
class Solution {
public:
    int numTrees(int n) {
        if(n == 1) return 1;
        vector<int> dp(n + 1);
        dp[0] = dp[1] = 1, dp[2] = 2;
        for(int i = 3; i <= n; ++i){
            for(int j = 0; j < i; ++j){
                dp[i] += dp[j] * dp[i - 1 - j];
            }
        }
        return dp[n];
    }
};
```

#### 39. 验证二叉搜索树 (leetcode-98)

题目：https://leetcode-cn.com/problems/validate-binary-search-tree/

思路：直接遍历二叉树即可。对于递归版本，需要左子树和右子树的最大最小值，用以检验是否满足二叉搜索树的性质；对于非递归版本就简单的多了，直接判断访问的节点是不是升序即可。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    // 1. 左子树的最大值 < root->val < 右子树的最小值
    // bool isValidBST(TreeNode* root) {
    //     long long maxn, minn;
    //     return isValidBST(root, maxn, minn);
    // }
    // bool isValidBST(TreeNode* root, long long& maxn, long long& minn){
    //     if(root == nullptr) return true;
    //     long long maxl = INT64_MIN, minl = INT64_MAX;
    //     long long maxr = INT64_MIN, minr = INT64_MAX;
    //     if(isValidBST(root->left, maxl, minl) && isValidBST(root->right, maxr, minr)){
    //         if(root->val > maxl && root->val < minr){
    //             maxr == INT64_MIN ? maxn = root->val : maxn = maxr;
    //             minl == INT64_MAX ? minn = root->val : minn = minl;
    //             return true;
    //         }
    //     }
    //     return false;
    // }
    // 2. 非递归中序遍历
    bool isValidBST(TreeNode* root){
        if(root == nullptr) return true;
        stack<TreeNode*> stack1;
        TreeNode* node = root;
        long long last = INT64_MIN;
        while(node || !stack1.empty()){
            while(node){
                stack1.push(node);
                node = node->left;
            }
            node = stack1.top(); stack1.pop();
            if(node->val <= last) return false;
            else last = node->val;
            node = node->right;
        }
        return true;
    }
};
```

#### 40. 对称二叉树 (leetcode-101)

题目：https://leetcode-cn.com/problems/symmetric-tree/

思路：递归检查是否对称

```cpp
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        if(root == nullptr) return true;
        return isSymmetric(root->left, root->right);
    }
private:
    bool isSymmetric(TreeNode* left, TreeNode* right){
        if(left == nullptr && right == nullptr) return true;
        else if(left == nullptr || right == nullptr) return false;
        else if(left->val != right->val) return false;
        else return isSymmetric(left->left, right->right) && isSymmetric(left->right, right->left);
    }
};
```

#### 41. 二叉树的层序遍历 (leetcode-102)

题目：https://leetcode-cn.com/problems/binary-tree-level-order-traversal/

思路：直接默写bfs层序遍历即可。

```cpp
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        if(root == nullptr) return ans;
        queue<TreeNode*> Q;
        Q.push(root);
        while(!Q.empty()){
            vector<int> temp;
            for(int i = Q.size(); i > 0; --i){
                TreeNode* node = Q.front(); Q.pop();
                temp.push_back(node->val);
                if(node->left) Q.push(node->left);
                if(node->right) Q.push(node->right);
            }
            ans.push_back(temp);
        }
        return ans;
    }
};
```

#### 42. 二叉树的最大深度 (leetcode-104)

题目：https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/submissions/

思路：直接递归即可。

```cpp
class Solution {
public:
    int maxDepth(TreeNode* root) {
       if(root == nullptr) return 0;
       return max(maxDepth(root->left), maxDepth(root->right)) + 1; 
    }
};
```

#### 43. 从前序和中序遍历序列构造二叉树 (leetcode-105)

题目：https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/

思路：前序遍历的第一个节点是根节点，而中序遍历的根节点将序列分为两个部分。

```cpp
class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        if(preorder.size() == 0) return nullptr;
        return buildTree(preorder, 0, inorder, 0, inorder.size());
    }
    TreeNode* buildTree(vector<int>& preorder, int pstart, vector<int>& inorder, int istart, int len){
        if(len <= 0) return nullptr;
        TreeNode* node = new TreeNode(preorder[pstart]);
        int curr = istart;
        while(inorder[curr] != preorder[pstart]) curr++;
        node->left = buildTree(preorder, pstart + 1, inorder, istart, curr- istart);
        node->right = buildTree(preorder, pstart + curr - istart + 1, inorder, curr + 1, len - 1 - curr + istart);
        return node;
    }
};
```

#### 44. 二叉树展开为链表 (leetcode-114)

题目：https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/

思路：递归展开，需要将左子树的最后一个节点指向右子树的根节点。

```cpp
class Solution {
public:
    void flatten(TreeNode* root){
        if(root == nullptr) return;
        flatten(root->left);
        flatten(root->right);
        if(root->left){
            TreeNode* node = root->left;
            while(node->right) node = node->right;
            node->right = root->right;
            root->right = root->left;
            root->left = nullptr;
        }
    }
};
```

#### 45. 买卖股票的最佳时机 (leetcode-121)

题目：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/

思路：这是leetcode买卖股票系列最简单的一个题，题目要求的是只能买卖一次股票。买卖股票系列都可以用动态规划解决。

状态定义：
```cpp
dp[i][0]    // 当前手上没有股票时能够获得的最大利润
dp[i][1]    // 当前手上有股票时能够获得的最大利润
```
状态转移：
```cpp
dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
dp[i][1] = max(dp[i - 1][1], -prices[i]);
```
其中，只能买卖一次就体现在这个`-prices[i]`这里，如果可以买卖多次，那么应该修改为：
```cpp
dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
```
经过状态压缩之后的完整代码：
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
        return dp0;
    }
};
```

#### 46. 二叉树的最大路径和 (leetcode-124)

题目：https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/

思路：这是一道经典面试题，通过递归可以解决，但需要注意一些细节。

```cpp
class Solution {
public:
    int ans = INT32_MIN;
    int dfs(TreeNode* root){
        if(root == nullptr) return 0;
        int left = dfs(root->left);
        int right = dfs(root->right);
        ans = max(ans, left + right + root->val);
        return max(0, root->val + max(left, right));
    }
    int maxPathSum(TreeNode* root) {
        dfs(root);
        return ans;
    }
};
```
#### 47. 最长连续序列 (leetcode-128)

题目：https://leetcode-cn.com/problems/longest-consecutive-sequence/

思路：哈希表或者并查集。

```cpp
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> set;
        for (const int& num : nums) {
            set.insert(num);
        }
        int ans = 0;
        for (const int& num : set) {
            if (!set.count(num - 1)) { // 不存在 num - 1
                int curr = num;
                while (set.count(curr + 1)) {
                    curr++;
                }
                ans = max(ans, curr - num + 1);
            }
        }
        return ans;           
    }
};
```

#### 48. 只出现一次的数字 (leetcode-136)

题目：https://leetcode-cn.com/problems/single-number/

思路：经典位运算

```cpp
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int ans = 0;
        for(const auto& n : nums) ans ^= n;
        return ans;
    }
};
```

#### 49. 单词拆分 (leetcode-139)

题目：https://leetcode-cn.com/problems/word-break/solution/

思路：动态规划题目

```cpp
    bool wordBreak(string s, vector<string>& wordDict) {
        vector<bool> dp(s.size() + 1, false);
        dp[0] = true;
        unordered_map<string, int> Map;
        for(const string& word : wordDict){
            Map[word] = 1;
        }
        for(int i = 1; i <= s.size(); ++i){
            for(int j = 0; j < i; ++j){
                dp[i] = Map.count(s.substr(j, i - j)) && dp[j];
                if(dp[i]) break;
            }
        }
        return dp[s.size()];
    }
```

#### 50. 环形链表 (leetcode-141)

题目：https://leetcode-cn.com/problems/linked-list-cycle/

思路：快慢指针法

```cpp
class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode* fast = head, *slow = head;
        while(fast && fast->next){
            fast = fast->next->next;
            slow = slow->next;
            if(fast == slow) return true;
        }
        return false;
    }
};
```

#### 51. 环形链表II (leetcode-142)

题目：https://leetcode-cn.com/problems/linked-list-cycle-ii/

思路：快慢指针来判断是否有环，并计算环的长度。然后再用前后指针的方法来找到环的入口。

```cpp
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode* fast = head, *slow = head;
        while(fast && fast->next){
            fast = fast->next->next;
            slow = slow->next;
            if(fast == slow){ // 存在环
                int len = 1;
                slow = slow->next;
                while(slow != fast){
                    slow = slow->next;
                    len++; // 计算环的长度
                }
                slow = fast = head;
                for(int i = 0; i < len; ++i) fast = fast->next;
                while(slow != fast){
                    fast = fast->next;
                    slow = slow->next;
                }
                return slow; // 环的入口
            }
        }
        return nullptr;
    }
};
```

#### 52. LRU缓存机制 (leetcode-146)

题目：https://leetcode-cn.com/problems/lru-cache/

思路：LRU缓存可以利用`LinkedHashMap`来实现，但是`C++`中不存在该数据结构，因此可以使用哈希表+双向链表来实现。

```cpp
class LRUCache {
public:
    // 哈希表 + 链表
    LRUCache(int capacity) {
        Head = new Node(0, 0);
        Tail = new Node(0, 0);
        Head->next = Tail;
        Tail->prev = Head;
        cap = capacity;
        size = 0;
    }
    
    int get(int key) {
        Node* node1;
        node1 = Head->next;

        if(!Map.count(key)) return -1;
        Node* node = Map[key];

        node->prev->next = node->next;
        node->next->prev = node->prev;

        node1 = Head->next;
        node->next = Tail;
        Tail->prev->next = node;
        node->prev = Tail->prev;
        Tail->prev = node;
        return node->val;
    }
    
    void put(int key, int value) {
        Node* node = nullptr;
        // 找
        if(Map.count(key)){
            node = Map[key];
            node->val = value;
            node->prev->next = node->next;
            node->next->prev = node->prev;
        }else{
            node = new Node(key, value);
            Map[key] = node;
            size++;
        }
        // 插
        node->next = Tail;
        Tail->prev->next = node;
        node->prev = Tail->prev;
        Tail->prev = node;
        // 删
        if(size > cap){
            Node* temp = Head->next;
            Head->next = temp->next;
            temp->next->prev = Head;
            Map.erase(temp->key);
            delete temp;
        }
    }
private:
    struct Node{
        int key;
        int val;
        Node* next;
        Node* prev;
        Node(int k, int v) : val(v), key(k) {}
    };
    int cap;
    int size;
    Node* Head, *Tail;
    unordered_map<int, Node*> Map;
};
```

#### 53. 排序链表 (leetcode-148)

题目：https://leetcode-cn.com/problems/sort-list/

思路：题目要求时间复杂度不超过`O(nlogn)`，因此可以用归并排序或者是快排进行排序。

快排：
```cpp
class Solution {
public:
    unordered_map<ListNode*, ListNode*> prev;
    ListNode* partition(ListNode* start, ListNode* end){
        ListNode* pos = start;
        ListNode* p = start;
        while(p != end){
            if(p->val < end->val){
                swap(p->val, pos->val);
                pos = pos->next;
            }
            p = p->next;
        }
        swap(pos->val, end->val);
        return pos;
    }
    void quick_sort(ListNode* start, ListNode* end){
        if(start == nullptr || end == nullptr || start == end || end->next == start) return;
        ListNode* pos = partition(start, end);
        quick_sort(start, prev[pos]);
        quick_sort(pos->next, end);
    }
    ListNode* sortList(ListNode* head) {
        if(head == nullptr) return nullptr;
        ListNode* tail = head;
        prev[head] == nullptr;
        while(tail->next){
            prev[tail->next] = tail;
            tail = tail->next;
        } 
        quick_sort(head, tail);
        return head;
    }
};
```
归并排序：
```cpp
class Solution {
public:
    ListNode* merge(ListNode* l1, ListNode* l2) {
        if(l1 == nullptr) return l2;
        if(l2 == nullptr) return l1;
        if(l1->val < l2->val){
            l1->next = merge(l1->next, l2);
            return l1;
        }else{
            l2->next = merge(l1, l2->next);
            return l2;
        }
        return nullptr;
    }
    ListNode* sortList(ListNode* head) {
        if(head == nullptr || head->next == nullptr) return head;
        ListNode* slow = head, *fast = head, *pre = head;
        while(fast != nullptr && fast->next != nullptr) {
            pre = slow;
            slow = slow->next;
            fast = fast->next->next;
        }
        pre->next = nullptr;
        return merge(sortList(head), sortList(slow));
    }
};
```

#### 54. 乘积最大子数组 (leetcode-152)

题目：https://leetcode-cn.com/problems/maximum-product-subarray/

思路：假设以元素`i`结尾的子数组的最大乘积为`product[i]`。当`nums[i]`为正数时，`product[i]`与`i-1`结尾的子数组的最大乘积有关；当`nums[i]`为负数时，`product[i]`与`i-1`结尾的子数组的最小乘积有关；因此可以用动态规划求解：

状态定义：
```cpp
dp[i][0]    // 以nums[i]结尾的子数组的最小乘积
dp[i][1]    // 以nums[i]结尾的子数组的最大乘积
```

状态转换：
```cpp
if nums[i] > 0:
    dp[i][0] = min(nums[i], dp[i - 1][0] * nums[i]);
    dp[i][1] = max(nums[i], dp[i - 1][1] * nums[i]);
else:
    dp[i][0] = min(nums[i], dp[i - 1][1] * nums[i]));
    dp[i][1] = max(nums[i], dp[i - 1][0] * nums[i]));
```

完整代码：
```cpp
class Solution {
public:
    int maxProduct(vector<int>& nums){
        vector<vector<int>> dp(nums.size(), vector<int>(2, 0));
        dp[0][0] = dp[0][1] = nums[0];
        int ans = nums[0];
        for(int i = 1; i < nums.size(); ++i){
            if(nums[i] > 0){
                dp[i][0] = min(nums[i], dp[i - 1][0] * nums[i]);
                dp[i][1] = max(nums[i], dp[i - 1][1] * nums[i]);
            }else{
                dp[i][0] = min(nums[i], dp[i - 1][1] * nums[i]);
                dp[i][1] = max(nums[i], dp[i - 1][0] * nums[i]);
            }
            ans = max(ans, dp[i][1]);
        }
        return ans;
    }
};
```

#### 55. 最小栈 (leetcode-155)

题目：https://leetcode-cn.com/problems/min-stack/

思路：维护两个栈，一个保存`push`的元素，另一个保存当前栈中元素的最小值。

```cpp
class MinStack {
    stack<int> S1, S2;
public:
    /** initialize your data structure here. */
    MinStack() {
    }
    void push(int x) {
        S1.push(x);
        if(S2.empty()) S2.push(x);
        else{
            x < S2.top() ? S2.push(x) : S2.push(S2.top());
        }
    }
    void pop() {
        if(!S1.empty() && !S2.empty()){
            S1.pop(); S2.pop();
        }
    }
    int top() {
        return S1.top();
    }
    int getMin() {
        return S2.top();
    }
};
```

#### 56. 相交链表 (leetcode-160)

题目：https://leetcode-cn.com/problems/intersection-of-two-linked-lists/

思路：先计算两个链表的长度，然后计算出长度差`s`。最后两两个指针先后（`s`个节点）从两个链表开始前进，当两个指针相交点就是链表的交点。

```cpp
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        int cntA = 0, cntB = 0;
        ListNode* p = headA, *q = headB;
        while(p){
            p = p->next;
            cntA++;
        }
        while(q){
            q = q->next;
            cntB++;
        }
        p = headA; q = headB;
        if(cntA > cntB){
            for(int i = 0; i < cntA - cntB; ++i){
                p = p->next;
            }
        }else{
            for(int i = 0; i < cntB - cntA; ++i){
                q = q->next;
            }
        }
        while(p != q){
            if(p == nullptr || q == nullptr) return nullptr;
            p = p->next;
            q = q->next;
        }
        return p;
    }
};
```

#### 57. 多数元素 (leetcode-169)

题目：https://leetcode-cn.com/problems/majority-element/

思路：最直观的方法是利用哈希表计数，直接输出数量最多的元素即可。

```cpp
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        unordered_map<int, int> count;
        int ans = 0, maxcnt = 0;;
        for(const auto& num : nums){
            count[num]++;
            if(count[num] > maxcnt){
                maxcnt = count[num];
                ans = num;
            }
        }
        return ans;
    }
};
```

#### 58. 打家劫舍 (leetcode-198)

题目：https://leetcode-cn.com/problems/house-robber/

思路：经典动态规划问题。

状态定义：
```cpp
dp[i][0]    // 不偷当前房屋能够获得的最大金额
dp[i][1]    // 偷当前房屋能够获得的最大金额
```

状态转换：
```cpp
dp[i][0] = max(dp[i - 1][0], dp[i - 1][1]);
dp[i][1] = dp[i - 1][0] + nums[i];
```

完整代码：
```cpp
class Solution {
public:
    int rob(vector<int>& nums) {
        if(nums.size() == 0) return 0;
        vector<vector<int>> dp(nums.size(), vector<int>(2, 0));
        dp[0][0] = 0; dp[0][1] = nums[0];
        for(int i = 1; i < nums.size(); ++i){
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][01]);
            dp[i][1] = dp[i - 1][0] + nums[i];
        }
        return max(dp.back()[0], dp.back()[1]);
    }
};
```
注意：可以进行状态压缩。

#### 59. 岛屿数量 (leetcode-200)

题目：https://leetcode-cn.com/problems/number-of-islands/

思路：典型的dfs问题。

```cpp
class Solution {
public:
    vector<int> dx = {-1, 1, 0, 0};
    vector<int> dy = {0, 0, 1, -1};
    void dfs(vector<vector<char>>& grid, int x, int y){
        if(x < 0 || x >= grid.size() || y < 0 || y >= grid[0].size() || grid[x][y] == '0') return;
        grid[x][y] = '0';
        for(int i = 0; i < 4; ++i){
            dfs(grid, x + dx[i], y + dy[i]);
        }
    }
    int numIslands(vector<vector<char>>& grid) {
        if(grid.size() == 0 || grid[0].size() == 0) return 0;
        int cnt = 0;
        for(int x = 0; x < grid.size(); ++x){
            for(int y = 0; y < grid[0].size(); ++y){
                if(grid[x][y] == '1'){
                    cnt++;
                    dfs(grid, x, y);
                }
            }
        }
        return cnt;
    }
};
```

#### 60. 翻转链表 (leetcode-206)

题目：https://leetcode-cn.com/problems/reverse-linked-list/

思路：三指针法翻转链表

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(head == nullptr || head->next == nullptr) return head;
        ListNode* prev = nullptr, *curr = head, *next = head->next;
        while(curr){
            curr->next = prev;
            prev = curr;
            curr = next;
            if(next) next = next->next;
        }
        return prev;
    }
};
```

#### 61. 课程表 (leetcode-207)

题目：https://leetcode-cn.com/problems/course-schedule/

思路：dfs判断是否有环即可。此外，还可以用bfs、拓扑排序等方法。

```cpp
class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        uint8_t* vis = new uint8_t[numCourses]();
        vector<vector<int>> From(numCourses, vector<int>());
        // 建图
        for(int i = 0; i < prerequisites.size(); ++i){
            From[prerequisites[i][0]].push_back(prerequisites[i][1]);
        }
        for(int i = 0; i < numCourses; ++i){
            if(vis[i] == 0){
                vis[i] = 1;
                for(int j = 0; j < From[i].size(); ++j){
                    if(!dfs(From[i][j], From, vis)) return false;
                }
                vis[i] = 2;
            } 

        }
        return true;
    }
    bool dfs(int curr, vector<vector<int>>& From, uint8_t* vis){
        if(vis[curr] == 1) return false;
        if(vis[curr] == 2) return true;
        vis[curr] = 1;  // vis == 0 未访问过    vis == 1 正在访问   vis == 2 已访问过
        for(int i = 0; i < From[curr].size(); ++i){
            if(!dfs(From[curr][i], From, vis)) return false;
        }
        vis[curr] = 2;
        return true;
    }
};
```

#### 62. 实现前缀树 (leetcode-208)

题目：https://leetcode-cn.com/problems/implement-trie-prefix-tree/

思路：边插入边建立前缀树

```cpp
class Trie {
public:
    struct Node{
        Node(){
            Next = new Node*[26]();
        }
        bool exist = false;
        Node** Next;
    };
    Node* Head;
    /** Initialize your data structure here. */
    Trie() {
        Head = new Node;
    }
    
    /** Inserts a word into the trie. */
    void insert(string word) {
        Node* node = Head;
        for(const char& ch : word){
            if(node->Next[ch - 'a'] == nullptr) node->Next[ch - 'a'] = new Node;
            node = node->Next[ch - 'a'];
        }    
        node->exist = true;

    }
    /** Returns if the word is in the trie. */
    bool search(string word) {
        Node* node = Head;
        for(const char& ch : word){
            if(node->Next[ch - 'a'] == nullptr) return false;
            node = node->Next[ch - 'a'];
        }
        if(node->exist) return true;
        return false;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
        Node* node = Head;
        for(const char& ch : prefix){
            if(node->Next[ch - 'a'] == nullptr) return false;
            node = node->Next[ch - 'a'];
        }
        return true;
    }
    
};
```

#### 63. 数组中的第K个最大元素 (leetcode-215)

题目：https://leetcode-cn.com/problems/kth-largest-element-in-an-array/

思路：最大堆或者快速选择

```cpp
class Solution {
public:
    // 1. Priority queue
    // int findKthLargest(vector<int>& nums, int k) {
    //     priority_queue<int, vector<int>, greater<int>> Q;
    //     for(const int& num : nums){
    //         if(Q.size() < k) Q.push(num);
    //         else{
    //             if(Q.top() >= num) continue;
    //             Q.pop();Q.push(num);
    //         }
    //     }
    //     return Q.top();
    // }
    // 2. quick_select
    int partition(vector<int>& nums, int start, int end){
        int pivot = nums[end];
        int pos = start;
        for(int i = start; i < end; ++i){
            if(nums[i] < pivot){
                swap(nums[pos++], nums[i]);
            }
        }
        swap(nums[pos], nums[end]);
        return pos;
    }

    int findKthLargest(vector<int>& nums, int k){
        int start = 0, end = nums.size() - 1;
        int target = nums.size() - k;
        while(1){
            int pos = partition(nums, start, end);
            if(pos == target) return nums[pos];
            if(pos < target) start = pos + 1;
            if(pos > target) end = pos - 1;
        }
        return 0;    
    }
};
```

#### 64. 最大正方形 (leetcode-221)

题目：https://leetcode-cn.com/problems/maximal-square/

思路：这个题和之前的最大矩形很类似，但是相较于那个题会简单不少，因为正方形用动态规划会更简单。

状态定义：`dp[i][j]`是以`matrix[i][j]`为右下角的最大正方形的边长。

状态转移：
```cpp
dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1;
```

完整代码：
```cpp
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        if(matrix.size() == 0 || matrix[0].size() == 0) return 0; 
        vector<vector<int>> dp(matrix.size(), vector<int>(matrix[0].size()));
        int maxlen = 0;
        for(int i = 0; i < matrix.size(); ++i) if(matrix[i][0] == '1') {
            dp[i][0] = 1;
            maxlen = 1;
        }
        for(int i = 0; i < matrix[0].size(); ++i) if(matrix[0][i] == '1'){
            dp[0][i] = 1;
            maxlen = 1;
        }
        for(int i = 1; i < matrix.size(); ++i){
            for(int j = 1; j < matrix[0].size(); ++j){
                if(matrix[i][j] == '1'){
                    dp[i][j] = min(dp[i - 1][j], min(dp[i][j - 1], dp[i - 1][j - 1])) + 1;
                    maxlen = max(maxlen, dp[i][j]);
                } 
                   
            }
        }
        return maxlen * maxlen;

    }
};
```

#### 65. 翻转二叉树 (leetcode-226)

题目：https://leetcode-cn.com/problems/invert-binary-tree/

思路：递归翻转

```cpp
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if(root == nullptr) return root;
        swap(root->left, root->right);
        invertTree(root->left);
        invertTree(root->right);
        return root;
    }
};
```

#### 66. 回文链表 (leetcode-234)

题目：https://leetcode-cn.com/problems/palindrome-linked-list/

思路：首先用快慢指针找到中间节点，然后将后半链表反转，最后从链表两头向中间遍历，判断经过的节点是否相等。

```cpp
class Solution {
public:
    ListNode* reverse(ListNode* head){
        if(head == nullptr) return head;
        ListNode* prev = nullptr, *curr = head, *next = head->next;
        while(curr){
            curr->next = prev;
            prev = curr;
            curr = next;
            if(next) next = next->next;
        }
        return prev;
    }
    bool isPalindrome(ListNode* head) {
        if(head == nullptr) return true;
        int len = 0;
        ListNode* node = head;
        while(node != nullptr){
            node = node->next;
            len++;
        }
        node = head;
        for(int i = 0; i < len / 2; ++i){
            node = node->next;
        }
        ListNode* head1 = reverse(node);
        for(int i = 0; i < len / 2; ++i){
            if(head->val != head1->val) return false;
            head = head->next;
            head1 = head1->next;
        }
        return true;
    }
};
```

#### 67. 二叉树的最近公共祖先 (leetcode-236)

题目：https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/

思路：递归。当左子树中存在、右子树中存在、当前节点是目标节点之一三个条件中满足两个条件就是要求的最近公共祖先。

```cpp
class Solution {
public:
    TreeNode* ans;
    bool dfs(TreeNode* curr, TreeNode* p, TreeNode* q){
        if(curr == nullptr) return false;
        int cnt = dfs(curr->left, p, q) + dfs(curr->right, p, q) + (curr == p || curr == q);
        if(cnt == 2) ans = curr;
        return cnt;
    }
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        ans = root;
        dfs(root, p, q);
        return ans;
    }
};
```

#### 68. 除自身以外数组的乘积 (leetcode-238)

题目：https://leetcode-cn.com/problems/product-of-array-except-self/

思路：构造左右乘积数组。

```cpp
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int> L(nums.size(), 1);
        vector<int> R(nums.size(), 1);
        vector<int> res(nums.size());
        for(int i = 1; i < nums.size(); ++i){
            L[i] = L[i - 1] * nums[i - 1];
        }
        for(int i = nums.size() - 2; i >= 0; --i){
            R[i] = R[i + 1] * nums[i + 1];
        }
        res[0] = R[0];
        for(int i = 1; i < nums.size() - 1; ++i){
            res[i] = L[i] * R[i];
        }
        res[nums.size() - 1] = L[nums.size() - 1];
        return res;
    }
};
```

#### 69. 滑动窗口最大值 (leetcode-239)

题目：https://leetcode-cn.com/problems/sliding-window-maximum/

思路：类似于单调栈，这里可以使用`deque`来构造单调队列。

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

##### 70. 搜索二维矩阵II (leetcode-240)

题目：https://leetcode-cn.com/problems/search-a-2d-matrix-ii/

思路：从左上角或者右下角开始搜索。

```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        if(matrix.size() == 0 || matrix[0].size() == 0) return false;
        int row = 0, col = matrix[0].size() - 1;
        while(1){
            if(matrix[row][col] == target) return true;
            else if(matrix[row][col] < target) row++;
            else col--;
            if(col < 0 || row == matrix.size()) return false;
        } 
        return true;
    }
};
```

#### 71. 完全平方数 (leetcode-279)

题目：https://leetcode-cn.com/submissions/detail/79312874/

思路：动态规划或者`bfs`。

```cpp
class Solution {
public:
	int numSquares(int n) {
		vector<int> dp(n + 1, INT32_MAX - 1);
		dp[0] = 0, dp[1] = 1;
		for (int i = 2; i <= n; ++i) {
			for(int j = 1; j * j <= i; ++j)
                dp[i] = min(dp[i], 1 + dp[i - j * j]);
		}
		return dp[n];
	}
};
```
```cpp
class Solution {
public:
    int numSquares(int n) {
        vector<int> square;
        for(int i = 1;; ++i){
            square.push_back(i * i);
            if(square.back() > n) break;
        }
        queue<int> Q;
        Q.push(n);
        int res = 0;
        while(!Q.empty()){
            res++;
            for(int i = Q.size() - 1; i >= 0; --i){
                int curr = Q.front(); Q.pop();
                for(int j = 0; j < square.size(); ++j){
                    if(curr == square[j]) return res;
                    if(curr < square[j]) break;
                    Q.push(curr - square[j]);
                }
            }
        }
        return res;
    }
};
```

#### 72. 移动零 (leetcode-283)

题目：https://leetcode-cn.com/problems/move-zeroes/

思路：非零元素前移，后面补零。

```cpp
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int pos = 0;
        for(const auto& num : nums){
            if(num) nums[pos++] = num;
        }
        for(int i = pos; i < nums.size(); ++i){
            nums[i] = 0;
        }
    }
};
```

#### 73. 寻找重复数 (leetcode-287)

题目：https://leetcode-cn.com/problems/find-the-duplicate-number/

思路：题目要求使用`O(1)`空间，且时间复杂度小于`O(n2)`，因此很自然想到二分查找。

```cpp
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int low = 1, high = nums.size();
        while(low < high){
            int mid = low + ((high - low) >> 1);
            int cnt = 0;
            for(int i = 0; i < nums.size(); ++i){
                if(nums[i] <= mid) cnt++;
            }
            if(cnt <= mid) low = mid + 1;
            else high = mid;
        }
        return high;
    }
};
```
注意：当`nums[i] <= mid`的数小于等于`mid`时，表示`mid`过小。而当`cnt > mid`时，表示`mid >= target`。

#### 74. 二叉树的序列化和反序列化 (leetcode-297)

题目：https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/

思路：BFS层序序列化方便以迭代的方式反序列化。

```cpp
class Codec {
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        string ans;
        queue<TreeNode*> Q;
        Q.push(root);
        while(!Q.empty()){
            for(int i = Q.size() - 1; i >= 0; --i){
                TreeNode* curr = Q.front(); Q.pop();
                curr ? ans += (to_string(curr->val) + ",") : ans += "#,";
                if(curr){
                    Q.push(curr->left);
                    Q.push(curr->right);
                }
            }
        }
        return ans;
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        vector<string> split;
        int left = 0;
        for(int right = 0; right < data.size(); ++right){
            if(data[right] == ','){
                split.push_back(data.substr(left, right - left));
                left = right + 1;
            }
        }
        if(split[0] == "#") return nullptr;
        TreeNode* root = new TreeNode(atoi(split[0].c_str()));
        queue<TreeNode*> Q;
        Q.push(root);
        for(int i = 1; i < split.size(); i += 2){
            TreeNode* curr = Q.front(); Q.pop();
            if(split[i] != "#"){
                curr->left = new TreeNode(atoi(split[i].c_str()));
                Q.push(curr->left);
            }
            if(split[i + 1] != "#"){
                curr->right = new TreeNode(atoi(split[i + 1].c_str()));
                Q.push(curr->right);
            }
        }
        return root;
    }
};
```

#### 75. 最长上升子序列 (leetcode-300)

题目：https://leetcode-cn.com/problems/longest-increasing-subsequence/

思路：动态规划、贪心+二分查找

状态定义：`dp[i]`表示以`nums[i]`结尾的最长上升子序列。

状态转移:
```cpp
dp[i] = 1 + max(dp[j]) for j in [0 .. i-1]
```

```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        if(nums.size() == 0) return 0;
        vector<int> dp(nums.size(), 1);
        for(int i = 1; i < nums.size(); ++i){
            for(int j = 0; j < i; ++j){
                if(nums[j] < nums[i]) dp[i] = max(dp[i], dp[j] + 1);
            }
        }
        return *max_element(dp.begin(), dp.end());
    }
};
```

#### 76. 删除无效的括号 (leetcode-301)

题目：https://leetcode-cn.com/problems/remove-invalid-parentheses/

思路：官方题解给出的思路是回溯法，但我觉得BFS更容易理解，相当于找删除括号使其有效的最短路径。

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

#### 77. 最佳买卖股票时期含冷冻期 (leetcode-309)

题目：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/

思路：动态规划经典题。

状态定义：
```cpp
dp[i][0]    // 第i天手上没有股票最大利润
dp[i][1]    // 第i天手上有股票最大利润
dp[i][2]    // 第i天处于冷冻期最大利润
```

状态转换：
```cpp
dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
dp[i][1] = max(dp[i - 1][1], dp[i - 1][2] - prices[i]);
dp[i][2] = dp[i - 1][0];
```

完整代码：
```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if(prices.size() == 0) return 0;
        vector<vector<int>> dp(prices.size(), vector<int>(3, 0));
        dp[0][0] = dp[0][2] = 0; dp[0][1] = -prices[0];
        for(int i = 1; i < prices.size(); ++i){
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][2] - prices[i]);
            dp[i][2] = dp[i - 1][0];
        }
        return dp.back()[0];
    }
};
```

#### 78. 戳气球 (leetcode-312)

题目：https://leetcode-cn.com/problems/burst-balloons/

思路：这个题是hard，却是比较难。用动态规划可以求解，但是状态的定义比较难：

状态定义：`dp[i][j]`表示表示戳破`i`和`j`之间（开区间）的气球能够获得的最高分数。

状态转移：
```cpp
dp[i][j] = max(dp[i][k] + dp[k][j] + points[i] * points[k] * points[k]) for in range (i, j)
```

完整代码：
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

79. 零钱兑换 (leetcode-322)

题目：https://leetcode-cn.com/problems/coin-change/

思路：经典动态规划

状态定义：`dp[i]`表示兑换金额`i`所需的最少硬币数。

状态转换：
```cpp
dp[i] = 1 + max(dp[i - coin]) for coin in coins
```

完整代码：
```cpp
class Solution {
public:
	int coinChange(vector<int>& coins, int amount) {
        vector<int> dp(amount + 1, amount +1);
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

#### 80. 打家劫舍III (leetcode-337)

题目：https://leetcode-cn.com/problems/house-robber-iii/

思路：这个题的结构是树状，因此自底向上的动态规划比较麻烦，可以使用自顶向下+备忘的动态规划。

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
其中，`map0[node]`表示对于节点`node`，不偷窃该节点时，可以在该子树偷到的最大金额数。`map1[node]`同理。

#### 81. 比特位计数 (leetcode-338)

题目：https://leetcode-cn.com/problems/counting-bits/

思路：动态规划。对于数`i`，找到其最高位并置零得到`j`，则其1的个数为`1 + dp[j]`。

```cpp
class Solution {
public:
    vector<int> countBits(int num) {
        if(num == 0) return {0};
        vector<int> dp(num + 1, 0);
        dp[0] = 0; dp[1] = 1;
        for(int i = 2; i <= num; ++i){
            for(int j = 31; j > 0; --j){
                if(i & (1 << j)){
                    dp[i] = 1 + dp[i &~ (1 << j)];
                    break;
                }
            }
        }
        return dp;
    }
};
```

#### 82. 前K个高频元素 (leetcode-347)

题目：https://leetcode-cn.com/problems/top-k-frequent-elements/

思路：要求时间复杂度为`O(nlogn)`，因此很容易想到堆/有限队列。此外，需要用哈希表来计算出现的频率。此外，用快排其实也能达到`O(nlogn)`的时间复杂度。

```cpp
class Solution{
public:
    vector<int> topKFrequent(vector<int>& nums, int k){
        unordered_map<int, int> cnt;
        for(const auto& num : nums) cnt[num]++;
        vector<pair<int, int>> vec;
        for(const auto pair : cnt){
            vec.push_back(pair);
        }
        sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) { return a.second > b.second;});
        vector<int> ans;
        for(int i = 0; i < k; ++i) ans.push_back(vec[i].first);
        return ans;
    }
};
```

#### 83. 字符串解码 (leetcode-394)

题目：https://leetcode-cn.com/problems/decode-string/

思路：递归求解。遇到`[`递归到下一层，遇到`]`返回到上一层。

```cpp
class Solution {
public:
    string dfs(string& s, int& curr){
        string temp, res;
        int times = 0;
        while(curr < s.size()){
            if(s[curr] == '['){
                string temp =  dfs(s, ++curr);
                if(times == 0) times = 1;
                for(int i = 0; i < times; ++i) res += temp;
                times = 0;
            }else if(s[curr] == ']'){
                curr++;
                return res;
            }else if(isdigit(s[curr])){
                times = times * 10 + s[curr++] - '0';
            }else{
                res.push_back(s[curr++]);
            }
        }
        return res;
    }
    string decodeString(string s) {
        int curr = 0;
        return dfs(s, curr);
    }
};
```

#### 84. 除法求值 (leetcode-399)

#### 85. 根据身高重建队列 (leetcode-406)

题目：https://leetcode-cn.com/problems/queue-reconstruction-by-height/

思路：有一个很重要的点：矮个子在前面高个子是看不到的，因此可以在高个子前面任意插矮个子。具体解法是首先按`h`降序排序，`k`升序排序，然后挨个按照`k`值进行插入。

```cpp
class Solution {
public:
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
        // 贪心
        sort(people.begin(), people.end(), [](const auto& a, const auto& b){
            if(a[0] == b[0]) return a[1] < b[1];
            return a[0] > b[0];
        });
        // 插入
        vector<vector<int>> ans;
        for(int i = 0; i < people.size(); ++i){
            ans.insert(ans.begin() + people[i][1], people[i]);
        }
        return ans;
    }
};
```

#### 86. 分割等和子集 (leetcode-416)

题目：https://leetcode-cn.com/problems/partition-equal-subset-sum/

思路：实际上就是求解能否找到一个子集满足其和为总和的一半。由于数组的大小可能达到200，因此显然不能通过枚举子集来求解。换个思路发现可以把这个看做背包问题求解。

状态定义：`dp[i][j]`表示对于前`i`个数，能否正好找到一个子集，满足其和为`j`。

状态转移：
```cpp
dp[i][j] = true if j == num[i]
dp[i][j] = dp[i - 1][j] for j < nums[i]
dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i]] for j > nums[i]
```
完整代码：
```cpp
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = accumulate(nums.begin(), nums.end(), 0);
        if(sum & 1) return false;
        int target = sum / 2;
        vector<vector<bool>> dp(nums.size(), vector<bool>(target + 1, false));
        if(nums[0] == target) dp[0][nums[0]] = true;
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

#### 87. 路径总和III (leetcode-437)

题目：https://leetcode-cn.com/problems/path-sum-iii/

思路：思路其实不难，难点在于如何更高效，下面是一个不高效但简洁的做法。

```cpp
class Solution {
public:
    int ans;
    void dfs(TreeNode* node, vector<int> sums, const int& sum){
        if(node == nullptr) return;
        sums.push_back(0);
        for(int i = 0; i < sums.size(); ++i){
            sums[i] += node->val;
            if(sums[i] == sum)
                ans ++;
        }        
        dfs(node->left, sums, sum);
        dfs(node->right, sums, sum);
    }
    int pathSum(TreeNode* root, int sum) {
        vector<int> sums;
        dfs(root, sums, sum);
        return ans;
    }
private:
}; 
```

#### 88. 找到字符串中所有字母异位词 (leetcode-438)

题目：https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/

思路：滑动窗口

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

#### 89. 找到所有数组中消失的数字 (leetcode-448)

题目：https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/submissions/

思路：递归的让所有元素归位，如让`3`归位到`nums[2]`，让`5`归位到`nums[4]`。最后遍历一边数组，如果`nums[i] != i + 1`，那么数组中肯定不存在`i + 1`。

```cpp
class Solution {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        for(int i = 0; i < nums.size(); ++i){
            while(nums[nums[i] - 1] != nums[i]){
                swap(nums[i], nums[nums[i] - 1]);
            }
        }
        vector<int> ans;
        for(int i = 0; i < nums.size(); ++i){
            if(nums[i] != (i + 1)) ans.push_back(i + 1);
        }
        return ans;
    }
};
```

#### 90. 汉明距离 (leetcode-461)

题目：https://leetcode-cn.com/problems/hamming-distance/

思路：先对两个数求异或，然后统计`1`的位数。

```cpp
class Solution {
public:
    int hammingDistance(int x, int y) {
        int res = x ^ y;
        int cnt = 0;
        for(int i = 0; i < 32; ++i){
            if(res & (1 << i)) cnt++;
        }
        return cnt;
    }
};
```

#### 91. 目标和 (leetcode-494)

题目：https://leetcode-cn.com/problems/target-sum/

思路：最简单的一个方法就是dfs，但是dfs的时间复杂度会达到O(2^N^)。另一种方法是动态规划，可以看做是0-1背包的问题。

状态定义：`dp[i][j]`表示前`i`个数能够构成`j`的组合数。

状态转换：
```cpp
dp[i][j] += dp[i - 1][j - nums[i]] if j - nums[i] >= 0
dp[i][j] += dp[i - 1][j + nums[i]] if j - nums[i] < rmax
```

完整代码：
```cpp
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int S) {
        const int maxlen = 1001;
        if(S >= maxlen) return 0;
        vector<vector<int>> dp(nums.size(), vector<int>(maxlen * 2, 0));
        dp[0][maxlen + nums[0]] ++;
        dp[0][maxlen - nums[0]] ++;
        for(int i = 1; i < nums.size(); ++i){
            for(int j = 0; j < maxlen * 2; ++j){
                if(j - nums[i] >= 0) dp[i][j] += dp[i - 1][j - nums[i]];
                if(j + nums[i] < 2 * maxlen) dp[i][j] += dp[i - 1][j + nums[i]];
            }
        }
        return dp[nums.size() - 1][maxlen + S];
    }
};
```

#### 92. 把二叉树转换为累加树 (leetcode-538)

题目：https://leetcode-cn.com/problems/convert-bst-to-greater-tree/

思路：逆序递归遍历

```cpp
class Solution {
public:
    int sum = 0;
    void dfs(TreeNode* root){
        if(root == nullptr) return;
        if(root->right) dfs(root->right);
        root->val += sum;
        sum = root->val;
        if(root->left) dfs(root->left);
    }
    TreeNode* convertBST(TreeNode* root) {
        dfs(root);
        return root;
    }
};
```

#### 93. 二叉树的直径 (leetcode-543)

题目：https://leetcode-cn.com/problems/diameter-of-binary-tree/

思路：递归

```cpp
class Solution {
public:
    int maxd = 0;
    int dfs(TreeNode* root, int depth){
        if(root == nullptr) return depth;
        int left = 0, right = 0;
        left = dfs(root->left, depth + 1);
        right = dfs(root->right, depth + 1);
        int dis = abs(left + right - 2 * depth - 2);
        if(dis > maxd) maxd = dis;
        return max(left, right);
    }
    int diameterOfBinaryTree(TreeNode* root) {
        dfs(root, 0);
        return maxd;
    }
};
```

#### 94. 和为K的子数组 (leetcode-560)

题目：https://leetcode-cn.com/problems/subarray-sum-equals-k/

思路：转换为前缀和之差为K的个数。

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

#### 95. 最短无序连续子数组 (leetcode-581)

题目：https://leetcode-cn.com/problems/shortest-unsorted-continuous-subarray/

思路：最简单的方法是直接排序比较差异，时间复杂度为`O(nlongn)`；另一种方法是使用栈，看了下图就明白了：

<img src = './95.png' width = '50%'>

```cpp
class Solution {
public:
    int findUnsortedSubarray(vector<int>& nums) {
        stack<int> S;
        int left = nums.size() - 1, right = 0;
        for (int i = 0; i < nums.size(); ++i) {
            while (!S.empty() && nums[S.top()] > nums[i]){
                left = min(left, S.top()); S.pop();
            }
            S.push(i);
        }
        while(!S.empty()) S.pop();
        for (int i = nums.size() - 1; i >= 0; --i) {
            while (!S.empty() && nums[S.top()] < nums[i]){
                right = max(right, S.top()); S.pop();
            }
            S.push(i);
        }
        return right > left ? right - left + 1 : 0;
    }
};
```

#### 96. 合并二叉树 (leetcode-617)

题目：https://leetcode-cn.com/problems/merge-two-binary-trees/

思路：递归合并

```cpp
class Solution {
public:
    TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
        if(t1 && t2){
            t1->val += t2->val;
            t1->left = mergeTrees(t1->left, t2->left);
            t1->right = mergeTrees(t1->right, t2->right);
        }else if(t2){
            return t2;
        }
        return t1;
    }
};
```

#### 97. 任务调度器 (leetcode-621)

题目：https://leetcode-cn.com/problems/task-scheduler/

思路：贪心算法 + 时间戳

```cpp
class Solution {
public:
    struct Task{
        int cnt = 0;
        int ts = -100;
    };
    int leastInterval(vector<char>& tasks, int n) {
        Task* Tasks = new Task[27];
        for(auto task : tasks) Tasks[task - 'A'].cnt++;
        sort(Tasks, Tasks + 27, [](const Task& a, const Task& b){
            return a.cnt > b.cnt;
        });
        int end = 0;
        while(Tasks[end].cnt) end++;
        int time = 0;
        while(1){
            if(Tasks[0].cnt == 0) break;
            for(int i = 0; i < end; ++i){
                if(Tasks[i].cnt > 0 && Tasks[i].ts + n < time){
                    Tasks[i].cnt--; Tasks[i].ts = time;
                    while(Tasks[i].cnt < Tasks[i + 1].cnt){
                        swap(Tasks[i], Tasks[i + 1]);
                        i++;
                    }
                    if(Tasks[i].cnt == 0) end = i; 
                    break;
                }
            }
            time++;
        }
        return time;
    }
};
```

#### 98. 回文子串 (leetcode-647)

题目：https://leetcode-cn.com/problems/palindromic-substrings/solution/

思路：暴力法、中心延伸法、动态规划。

```cpp
class Solution {
public:
    int countSubstrings(string s) {
        vector<vector<bool>> dp(s.size(), vector<bool>(s.size(), true));
        int cnt = s.size();
        for(int d = 1; d < s.size(); ++d){
            for(int i = 0; i < s.size() - d; ++i){
                if(s[i] == s[i + d] && dp[i + 1][i + d -1]) cnt ++;
                else dp[i][i + d] = false;
            }
        }
        return cnt;
    }
};
```

#### 99. 每日温度

题目：https://leetcode-cn.com/problems/daily-temperatures/

思路：最典型的单调栈题目，当然也可以用动态规划。

```cpp
// class Solution {
// public:
//     vector<int> dailyTemperatures(vector<int>& T) {
//         if(T.size() == 0) return {};
//         vector<int> dp(T.size(), 0);
//         for(int i = T.size() - 2; i >= 0; --i){
//             int pos = i + 1;
//             while(1){
//                 if(T[i] < T[pos]){
//                     dp[i] = pos - i; break;
//                 }
//                 if(dp[pos] == 0) break;
//                 pos += dp[pos];
//             }
//         }
//         return dp;
//     }
// };

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