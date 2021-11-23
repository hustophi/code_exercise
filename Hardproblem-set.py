#合并k个已排序的链表并将其作为一个已排序的链表返回。分析并描述其复杂度(O(nlogk),)
#
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# @param lists ListNode类一维数组 
# @return ListNode类
import heapq
class Solution:
    def mergeKLists(self , lists ):
        tmpNode = ListNode(-1)
        cur = tmpNode
        tmpList = [(lists[i].val, i) for i in range(len(lists)) if lists[i]]
        heapq.heapify(tmpList)      #trick:以(节点值,节点索引)建堆, 从而避免值相同时无法比较而无法建堆的问题
        while tmpList:
            val, idx = heapq.heappop(tmpList)
            cur.next = lists[idx]
            cur = cur.next
            if cur.next: 
                heapq.heappush(tmpList,(cur.next.val, idx))
                lists[idx] = cur.next      #更新lists第idx个节点
        return tmpNode.next
        # write code here
################################################
#假设你有一个数组prices，长度为n，其中prices[i]是某只股票在第i天的价格，请根据这个价格数组，返回买卖股票能获得的最大收益
#1. 你最多可以对该股票有两笔交易操作，一笔交易代表着一次买入与一次卖出，但是再次购买前必须卖出之前的股票
#2. 如果不能获取收益，请返回0
#3. 假设买入卖出均无手续费
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# 两次交易所能获得的最大收益
# @param prices int整型一维数组 股票每一天的价格
# @return int整型
#key:遍历数组，以当前天作为两次交易的分隔点求最大收益，两次的最大收益可分别用动态规划求
class Solution:
    def maxProfit(self , prices: List[int]) -> int:
        res = 0
        if not prices: return res
        lenth = len(prices)
        first = [0] * lenth     #first[i]为[0,i]内只进行一次交易的最大收益
        minimum = prices[0]
        for i in range(1,lenth):
            minimum = min(prices[i-1], minimum)  #[0,i-1]内的最低价格
            first[i] = max(prices[i] - minimum, first[i-1])  #分在第i天卖出和第i天前卖出讨论
        second = [0] * lenth    #second[i]为[i,lenth-1]内只进行一次交易的最大收益
        maximum = prices[-1]
        for i in range(lenth-2, -1, -1):
            maximum = max(maximum, prices[i+1]) #[i+1, lenth-1]的最高价格
            second[i] = max(maximum - prices[i], second[i+1]) #分在第i天买入和第i天后买入讨论
        for i in range(lenth):
            res = max(first[i] + second[i], res)
        return res
        # write code here
################################################
#二叉树里面的路径被定义为:从该树的任意节点出发，经过父=>子或者子=>父的连接，达到任意节点的序列。
#注意: 1.同一个节点在一条二叉树路径里中最多出现一次; 2.一条路径至少包含一个节点，且不一定经过根节点
#给定一个二叉树的根节点root，请你计算它的最大路径和
#key: 递归计算经过每个节点的最大路径和,维护一个最大值
#
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# @param root TreeNode类 
# @return int整型
#
class Solution:
    def maxPathSum(self , root ):
        res = -float('inf')  #初始化
        def help(root):      #IMPORTANT: help函数递归计算以该节点为根节点的子树中寻找以该节点为起点的一条路径,使得该路径上的节点值之和最大
            nonlocal res     #维护一个全局变量, 记录最大值
            if root:
                l = help(root.left)
                r = help(root.right)
                val = max(l, 0) + max(r, 0) + root.val  #val为经过该节点的最大路径和 (NOTES:此步可直接在help递归过程中得到,而无需在help外单独使用递归,可减少空间复杂度)
                if val > res:
                    res = val 
                return max(l, r, 0) + root.val
            else:
                return 0
        help(root)
        return res 
        # write code here        
################################################
#给出一个长度为 n 的，仅包含字符 '(' 和 ')' 的字符串，计算最长的格式正确的括号子串的长度
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可 
# @param s string字符串 
# @return int整型
#动态规划解法
#dp[i]为以s[i]结尾的最长正确括号子串长度,对s[i]和s[i-1]分情况
#注意到最长正确括号串的特殊性质知: 当s[i]==s[i-1]==')'时,与s[i]配对的只可能是s[i-1-dp[i-1]]
#以s[i]结尾的正确括号串必包含以s[i-1]结尾的正确括号串, 且长度至少增加2
#NOTES:若无以s[i-1]结尾的正确括号串,那么也不存在以s[i]结尾的正确括号串 (反证法)
class SolutionOfDp:
    def longestValidParentheses(self , s: str) -> int:
        L = len(s)
        if L <= 1: return 0
        dp = [0] * L
        dp[1] = 2 if s[:2] == '()' else 0
        res = max(dp[1], 0)   #维护最大值
        for i in range(2, L):
            if s[i] == ')':     #对s[i]分情况
                if s[i-1] == '(': dp[i] = dp[i-2] + 2     #对s[i-1]分情况
                else:
                    if i-1-dp[i-1] >= 0 and s[i-1-dp[i-1]] == '(': #判断是否有s[i]的配对字符
                        dp[i] = dp[i-1] + 2 + \
                        (dp[i-2-dp[i-1]] if i-2-dp[i-1] >= 0 else 0) #dp方程, s[i]配对后还应考虑能否与之前的串相连接
                    else: dp[i] = 0
            res = max(res, dp[i])
        return res
        # write code here
#栈解法 
#始终保持栈底元素为当前已经遍历过的元素中「最后一个没有被匹配的右括号的下标」,其他元素为依次加入的左括号的下标
#1、对于每个'(' ,将它的下标放入栈中
#2、对于每个')' ，先弹出栈顶元素表示匹配了当前右括号 (用下面的trick可保证遇到')'时栈定不空)：
#    如果栈为空，说明当前的右括号为没有被匹配的右括号，将其下标放入栈中来更新我们之前提到的「最后一个没有被匹配的右括号的下标」
#    如果栈不为空，当前右括号的下标减去栈顶元素即为「以该右括号为结尾的最长有效括号的长度」(这是由于此时的栈顶元素必是以以该右括号为结尾的最长有效括号子串的前一个字符下标)
#3、从前往后遍历字符串并更新答案即可
#trick:由于一开始栈为空,若第一个字符为左括号我们会将其放入栈中,就不满足提及的「最后一个没有被匹配的右括号的下标」,为了保持统一,我们在一开始的时候往栈中放入-1(相当于虚拟'('的下标)
 class SolutionOfStack:
    def longestValidParentheses(self , s ):
        stack = [-1]
        ans = 0
        for i, ch in enumerate(s):
            if ch == '(':
                # 左括号下标入栈
                stack.append(i)
            else:
                if len(stack) > 1:
                    # 匹配括号
                    stack.pop()
                    # 最大括号长度
                    ans = max(ans, i-stack[-1])
                else:
                    # 将其下标放入栈中
                    stack[-1] = i
        return ans
        # write code here
#双指针解法
#利用两个计数器 left 和 right
#从左到右遍历字符串: 对于每个'(',left+1, 对于每个')',right+1; 判断left与right大小, 若相等,计算当前有效字符串的长度: 2*right;若right > left, 将left,right置0
#IMPORTANT: 这样的做法贪心地考虑了以当前字符下标结尾的有效括号串长度 (KEY), 每次当右括号数量多于左括号数量的时候之前的字符都扔掉不再考虑, 重新从下一个字符开始计算
#但这样会漏掉一种情况：左括号的数量始终大于右括号的数量, 如((), 此时最长有效括号是求不出来的
#IMPORTANT: 只需要 再从右往左遍历用类似的方法计算即可,但此时判断条件反了过来
class SolutionOfDoubPointer:
    def longestValidParentheses(self , s: str) -> int:
        left = right = maxlength = 0
        for i in range(len(s)):
            if s[i] == '(':
                left += 1
            else:
                right += 1
            if left == right:
                maxlength = max(maxlength, 2 * right)
            elif right > left:
                left = right = 0
        left = right = 0
        for i in range(len(s)-1, -1, -1):
            if s[i] == '(':
                left += 1
            else:
                right += 1
            if left == right:
                maxlength = max(maxlength, 2 * left);
            elif left > right:
                left = right = 0
        return maxlength
        # write code here
################################################
#请实现支持'?'和'*'的通配符模式匹配，'?' 可以匹配任何单个字符(即长度为1)；'*' 可以匹配任何字符序列（包括空序列）
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可 
# @param s string字符串 
# @param p string字符串 
# @return bool布尔型
#KEY: dp[i][j]表示字符串s的前i个字符和模式串p的前j个字符是否能匹配，并对p的第j个字符分情况讨论得dp方程
class Solution:
    def isMatch(self , s: str, p: str) -> bool:
        lenthS, lenthP = len(s), len(p)
        dp = [[False] * (lenthP + 1) for i in range(lenthS + 1)]
        dp[0][0] = True
        for j in range(1, lenthP+1):
            if p[j-1] == '*': dp[0][j] = True    #边界条件
            else: break
        for i in range(1, lenthS+1):
            for j in range(1, lenthP+1):
                if s[i-1] == p[j-1] or p[j-1] == '?': dp[i][j] = dp[i-1][j-1]  #若p[j-1]是字符或?,那么对应的s[i-1]必须是小写字母才可能匹配
                elif p[j-1] == '*': 
                    dp[i][j] = dp[i-1][j] or dp[i][j-1]          #IMPORTANT: 如果p[j]是星号则:1.不使用这个星号，dp[i][j]=dp[i][j-1]转移过来; 2.使用这个星号,dp[i][j]=dp[i-1][j]
        return dp[-1][-1]
        # write code here
################################################
#给定两个字符串str1和str2，再给定三个整数ic，dc和rc，分别代表插入、删除和替换一个字符的代价，请输出将str1编辑成str2的最小代价，0≤ ic,dc,rc ≤10000
#
# min edit cost
# @param str1 string字符串 the string
# @param str2 string字符串 the string
# @param ic int整型 insert cost
# @param dc int整型 delete cost
# @param rc int整型 replace cost
# @return int整型
class Solution:
    def minEditCost(self , str1 , str2 , ic , dc , rc ):
        lenth1, lenth2 = len(str1), len(str2)
        dp = [[0] * (lenth2+1) for i in range(lenth1+1)]
        for i in range(1, lenth1+1):
            dp[i][0] = i * dc
        for j in range(1, lenth2+1):
            dp[0][j] = j * ic
        for i in range(1, lenth1+1):
            for j in range(1, lenth2+1):
                #对str1的最后一个位置的最后一次操作分类讨论
                if str1[i-1] == str2[j-1]: dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1] + rc, dp[i-1][j] + dc, dp[i][j-1] + ic)     #IMPORTANT
        return dp[-1][-1]
        # write code here
