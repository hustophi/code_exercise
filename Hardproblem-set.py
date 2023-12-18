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
################################################
#请实现两个函数，分别用来序列化和反序列化二叉树，要求能够根据序列化之后的字符串重新构造出一棵与原二叉树相同的树
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Serialize(self, root):
        if not root:
            return '#'
        return str(root.val)+','+self.Serialize(root.left)+','+self.Serialize(root.right)  #先序遍历,递归生成序列化字符串,None节点记为#
        # write code here
    def Deserialize(self, s):#反序列化
        list = s.split(',')#利用split和逗号分隔，并放入列表中
        return self.DeserializeTree(list)#直接返回列表的反序列化值
        # write code here
    def DeserializeTree(self, list):#建立反序列二叉树的函数
        if len(list) <= 0:
            return None
        val = list.pop(0)
        root = None
        if val != '#':#如果不是空节点，
            root = TreeNode(int(val))
            root.left = self.DeserializeTree(list)#左节点
            root.right = self.DeserializeTree(list)#右节点
        return root
################################################
#给定一棵树，求出这棵树（不一定是二叉树）的直径，即树上最远两点的距离
# class Interval:
#     def __init__(self, a=0, b=0):
#         self.start = a
#         self.end = b
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# 树的直径
# @param n int整型 树的节点个数
# @param Tree_edge Interval类一维数组 树的边
# @param Edge_value int整型一维数组 边的权值
# @return int整型
#IMPORTANT:在一个连通无向无环图中，以任意结点出发所能到达的最远结点，一定是该图直径的端点之一，故可以做两次DFS来计算树的直径
from collections import defaultdict
#defaultdict接受一个factory_function作为参数
#这个factory_function可以是list、set、str等,作用是当key不存在时,返回的是factory_function的默认值,如list对应[],str对应空字符串,set对应set(),int对应0
class Solution:
    def solve(self , n: int, Tree_edge: List[Interval], Edge_value: List[int]) -> int:
        graph = reconstruct(n, Tree_edge, Edge_value)
        used = [0] * n
        remote, _ = dfs(graph, 0, used)
        _, maxlen = dfs(graph, remote, used)
        find_path(graph, remote, used, 0, maxlen, path)
        return maxlen, [remote]+ maxPath     #返回直径和对应的路径
def reconstruct(n, Tree_edge, Edge_value):       #使用图的邻接表存储形式
    graph = defaultdict(dict)                                   #{vertex:{vertex:weight}}
    for i in range(len(Tree_edge)):
        v1, v2 , w = Tree_edge[i].start, Tree_edge[i].end, Edge_value[i]
        graph[v1][v2] = graph[v2][v1] = w
    return graph
def dfs(graph, iniVertex, used):  #回溯, 求离iniVertex最远的点及二者的距离
    remote, maxlen = iniVertex, 0 #初始化
    used[iniVertex] = 1           #used列表记录当前路径中已使用的节点, 避免冗余计算
    for v in graph[iniVertex]:
        if used[v]: continue
        used[v] = 1
        tmp_v, pathlen = dfs(graph, v, used)
        used[v] = 0               #探索完一种可能后还原
        if pathlen + graph[iniVertex][v] > maxlen:  #判断, 更新最远点即最远距离
            remote = tmp_v
            maxlen = pathlen + graph[iniVertex][v]
    used[iniVertex] = 0
    return remote, maxlen
def find_path(graph, vertex, used, tmpSum, target, path):
    global maxPath
    used[vertex] = 1
    for v in graph[vertex]:
        if used[v]: continue
        path.append(v)
        used[v] = 1
        tmp = tmpSum + graph[vertex][v]
        if tmp == target: maxPath = path.copy()
        find_path(graph, v, used, tmp, target, path)
        used[v] = 0
        path.pop()
        # write code here
#法2, 图的构建同法1
class Solution:
    def __init__(self):
        self.maxPath = 0
    def solve(self , n: int, Tree_edge: List[Interval], Edge_value: List[int]) -> int:
        graph = reconstruct(n, Tree_edge, Edge_value)
        used = [0] * n
        self.dfs(graph, 0, used)
        return self.maxPath
    def dfs(self, graph, iniVertex, used):  #回溯, 求离iniVertex最远的点及二者的距离
        used[iniVertex] = 1
        d1, d2 = 0, 0    #d1, d2分别为当前未遍历节点距iniVertex的最远距离和次远距离
        for v in graph[iniVertex]:
            if used[v]: continue
            used[v] = 1
            tmp = self.dfs(graph, v, used)
            tmp += graph[iniVertex][v]
            if tmp >= d1: d1, d2 = tmp, d1  #IMPORTANT: 更新d1, d2
            elif tmp > d2: d2 = tmp
            used[v] = 0
            self.maxPath = max(self.maxPath, d1 + d2)  #整棵树的直径为所有节点d1+d2的最大值
        used[iniVertex] = 0
        return d1
        # write code here
################################################
#给数独中的剩余的空格填写上数字, 空格用字符'.'表示 (假设给定的数独只有唯一的解法)
# @param board char字符型二维数组 
# @return void
#
import math
class Solution:
    def __init__(self):
        self.rows, self.cols, self.miniBoard = [], [], []
    def solveSudoku(self , board ):
        r = c = len(board)
        self.rows, self.cols, self.miniBoard = [set() for i in range(r)], [set() for i in range(r)], [set() for i in range(r)]
        remain = []
        for i in range(r):
            for j in range(c):
                if board[i][j] != '.':
                    self.rows[i].add(board[i][j])
                    self.cols[j].add(board[i][j])
                    idx = self.posToId(i, j, r)
                    self.miniBoard[idx].add(board[i][j])
                else: remain.append((i,j))
        self.dfs(board, c, remain, 0)
        return
    def dfs(self, board, c, remain, loc):       #判断当前board状态下remain[loc:]能否合法填完
        if loc == len(remain): return True
        sr, sc = remain[loc]
        for n in range(1, c+1):
            idx = self.posToId(sr, sc, c)
            if str(n) not in self.rows[sr] and str(n) not in self.cols[sc] and str(n) not in self.miniBoard[idx]:
                board[sr][sc] = str(n)
                self.rows[sr].add(str(n))
                self.cols[sc].add(str(n))
                self.miniBoard[idx].add(str(n))
                if not self.dfs(board, c, remain, loc+1):        #回溯
                    self.rows[sr].remove(str(n))
                    self.cols[sc].remove(str(n))
                    self.miniBoard[idx].remove(str(n))
                    board[sr][sc] = '.'
                else: return True                   #IMPORTANT
        return False
    def posToId(self, x, y, r):
        m = int(math.sqrt(r))
        idx = (x // m) * m + y // m
        return idx
        # write code here
################################################
#请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）
#在本题中，匹配是指字符串的所有字符匹配整个模式, str和pattern格式说明如下:
#1.str 可能为空，且只包含从 a-z 的小写字母。
#2.pattern 可能为空，且只包含从 a-z 的小写字母以及字符'.' 和'*'，无连续的'*'且'*'前必须有字符
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param str string字符串 
# @param pattern string字符串 
# @return bool布尔型
#dp[i][j]表示字符串str的前i个字符和模式串pattern的前j个字符是否能匹配
class Solution:
    def match(self , str: str, pattern: str) -> bool:
        if str and not pattern: return False
        lenS, lenP = len(str), len(pattern)
        dp = [[False] * (lenP+1) for i in range(lenS+1)]
        dp[0][0] = True
        for j in range(1, lenP+1):
            if  not j % 2 and pattern[j-1] == '*': dp[0][j] = True
        if str and (pattern[0] == '.' or pattern[0] == str[0]): dp[1][1] = True
        for i in range(1, lenS+1):
            for j in range(2, lenP+1):
                if pattern[j-1] != '*':
                    if pattern[j-1] == str[i-1] or pattern[j-1] == '.':
                        dp[i][j] = dp[i-1][j-1]
                #IMPORTANT: 当pattern[j-1] 为"*"的时候,如pattern[j-2:j]为 'b*',将其作为整体并分情况讨论
                else: 
                    if pattern[j-2] == '.' or pattern[j-2] == str[i-1]:
                        dp[i][j] = dp[i-1][j] or dp[i][j-2]       #KEY: 'b*' 匹配b至少1次 or 0次
                    else:
                        dp[i][j] = dp[i][j-2]
        return dp[-1][-1]
        # write code here
################################################
#地下城游戏：给定一个二维数组map，含义是一张地图
#游戏的规则如下:
#1）骑士从左上角出发，每次只能向右或向下走，最后到达右下角见到公主。
#2）地图中每个位置的值代表骑士要遭遇的事情。如果是负数，说明此处有怪兽，要让骑士损失血量。如果是非负数，代表此处有血瓶，能让骑士回血。
#3）骑士从左上角到右下角的过程中，走到任何一个位置时，血量都不能少于1。为了保证骑土能见到公主，初始血量至少是多少?
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可 
# @param mp int整型二维数组 
# @return int整型
#dp[i][j]的含义是如果骑士从(i,j)出发最后走到右下角，骑士至少应该具备的血量（注意：若设dp[i][j]表示(0,0)到(i,j)的最少血量，不易得转移方程）
#从右往左, 从下往上更新dp, 最终结果即为dp[0][0]
class Solution:
    def dnd(self , mp: List[List[int]]) -> int:
        dp = [[0] * len(mp[0]) for i in range(len(mp))]
        dp[-1][-1] = max(1-mp[-1][-1], 1)
        for i in range(len(mp)-2, -1, -1):
            dp[i][-1] = max(dp[i+1][-1]-mp[i][-1], 1)
        for j in range(len(mp[0])-2, -1, -1):
            dp[-1][j] = max(dp[-1][j+1]-mp[-1][j], 1)
        for i in range(len(mp)-2, -1, -1):
            for j in range(len(mp[0])-2, -1, -1):
                dp[i][j] = min(max(dp[i+1][j]-mp[i][j], 1), max(dp[i][j+1]-mp[i][j], 1))
        return dp[0][0]
        # write code here
################################################
# 给定一个长度为 n 的非负整数数组 num ，和一个整数 m ，你需要把这个数组 num 分成 m 个非空连续子数组。
# 请你找出这些连续子数组各自的和的最大值最小的方案并输出这个值。
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param num int整型一维数组 
# @param m int整型 
# @return int整型
#反向思考：以threshold将nums中的元素分割; 若分割组数大于m，说明threshold取小了；否则，说明threshold取大了, 因此可采用二分法寻找threshold
#threshold的取值范围：[max(nums), sum(nums)]
class Solution:
    def splitMin(self , num: List[int], m: int) -> int:
        # write code here
        left, right = max(num), sum(num)
        while left < right:
            mid = left + ((right - left) >> 1)
            if self.thresholdCutCnt(num, mid, m) > m: left = mid + 1
            else: right = mid
        return left
    def thresholdCutCnt(self, num, t, m):
        count, tmp = 1, 0
        for n in num:
            tmp += n
            if tmp > t:
                count += 1
                tmp = n
            if count > m: break
        return count
################################################
# 给定一个长度为 n 的正整数数组 coins，每个元素表示对应位置的金币数量。
# 取位置 i 的金币时，假设左边一堆金币数量是L，右边一堆金币数量为R，则获得L*cost[i]*R的积分。如果左边或右边没有金币，则金币数量视为1。
# 请你计算最多能得到多少积分。
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可 
# @param coins int整型一维数组 
# @return int整型
#
#设dp[i][j]表示从下标i~下标j的区间，所能获得的最大积分
#对于区间[i, j]:
#1. 若最后选取的数是左边界，即最后选取的是coins[i]
#   dp[i][j] = dp[i + 1][j] + coins[i - 1] * coins[i] * coins[j + 1]
#2. 最后选取的数是右边界，即最后选取的coins[j]
#   dp[i][j] = dp[i][j - 1] + coins[i - 1] * coins[j] * coins[j + 1]
#3. 最后选取的数是k，i < k < j
#   dp[i][j] = max(dp[i][j], dp[i][k - 1] + coins[i - 1] * coins[k] * coins[j + 1] + dp[k + 1][j])
#   时间复杂度：O(n^3)        空间复杂度：O(n^2)
class Solution:
    def getCoins(self , coins: List[int]) -> int:
        # write code here
        coins = [1] + coins + [1]
        dp = [[0] * len(coins) for i in range(len(coins))]
        dp[1][1] = coins[0] * coins[1]
        dp[-1][-1] = coins[-1] * coins[-2]
        for i in range(1, len(coins)-1):
            dp[i][i] = coins[i-1] * coins[i] * coins[i+1]
        for i in range(len(coins)-3, 0, -1):
            for j in range(i+1, len(coins)-1):
                tmp = 0
                for k in range(i+1, j):
                    tmp = max(tmp, dp[i][k-1] + dp[k+1][j] + coins[k]*coins[i-1]*coins[j+1])
                dp[i][j] = max(dp[i+1][j] + coins[i] * coins[i-1] * coins[j+1],
                                dp[i][j-1] + coins[j] * coins[j+1] * coins[i-1],
                                tmp)
        return dp[1][-2]
################################################
# 地上有一个 rows 行和 cols 列的方格。坐标从 [0,0] 到 [rows-1,cols-1] 。
# 一个机器人从坐标 [0,0] 的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于 threshold 的格子。
# 例如，当 threshold 为 18 时，机器人能够进入方格   [35,37] ，因为 3+5+3+7 = 18。
# 但是，它不能进入方格 [35,38] ，因为 3+5+3+8 = 19 。请问该机器人能够达到多少个格子？
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param threshold int整型 
# @param rows int整型 
# @param cols int整型 
# @return int整型
#
#动态规划：(i,j)可达等价于(i-1,j)或(i,j-1)可达且(i,j)数位和不超过threshold
class Solution:
    def movingCount(self , threshold: int, rows: int, cols: int) -> int:
        # write code here
        dp = [[False] * cols for i in range(rows)]
        dp[0][0] = True
        ret = 1
        for i in range(1, rows):
            dp[i][0] = dp[i-1][0] and self.digitSum(i) <= threshold
            ret += dp[i][0]
        for j in range(1, cols):
            dp[0][j] = dp[0][j-1] and self.digitSum(j) <= threshold
            ret += dp[0][j]
        for i in range(1, rows):
            for j in range(1,cols):        
                dp[i][j] = (dp[i-1][j] or dp[i][j-1]) and \
                                (self.digitSum(i) + self.digitSum(j) <= threshold)
                ret += dp[i][j]
        return ret
    
    def digitSum(self, a):
        s = 0
        while a:
            s += a % 10
            a //= 10
        return s
################################################
# 给定两个字符串 s 和 t ，请问 s 有多少个子序列等于 t 。
# s 的子序列指从 s 中删除任意位置任意个字符留下的字符串。结果对2**31取余
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param s string字符串 
# @param t string字符串 
# @return int整型
#
#dp[i][j]为s的前i个字符包含t的前j个字符的子序列个数
class Solution:
    def countSubseq(self , s: str, t: str) -> int:
        # write code here
        if len(s) < len(t): return 0
        dp = [[0] * (len(t)+1) for i in range(len(s)+1)]
        for i in range(len(s)+1):
            dp[i][0] = 1
        for i in range(1, len(s)+1):
            for j in range(1, len(t)+1):
                dp[i][j] = dp[i-1][j]
                if s[i-1] == t[j-1]: dp[i][j] += dp[i-1][j-1]
        return dp[-1][-1] % (2**31)
################################################
# 给定一个正整数 n 和一个正整数 k ，请你给出 [1,n] 的第 k 个排列。
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param n int整型 
# @param k int整型 
# @return string字符串
#
class Solution:
    def KthPermutation(self , n: int, k: int) -> str:
        # write code here
        self.k = k
        used = [0] * (n+1)
        tmp = []
        self.dfs(n, tmp, used)
        return ''.join(map(str, tmp))
    def dfs(self, n, tmp, used):
        if len(tmp) == n:
            self.k -= 1
            return
        for i in range(1, n+1):
            if used[i]: continue
            tmp.append(i)
            used[i] = 1
            self.dfs(n, tmp, used)
            if self.k == 0: return
            tmp.pop()
            used[i] = 0
################################################
# 帅帅经常跟同学玩一个矩阵取数游戏：对于一个给定的 n*m 的矩阵，矩阵中的每个元素 均为非负整数。游戏规则如下：
# 1.每次取数时须从每行各取走一个元素，共 n 个。m 次后取完矩阵所有元素；
# 2.每次取走的各个元素只能是该元素所在行的行首或行尾；
# 3.每次取数都有一个得分值，为每行取数的得分之和，每行取数的得分 = 被取走的元素值 * 2i，其中i表示第 i 次取数（从1开始编号）；
# 4.游戏结束总得分为 m 次取数得分之和。
# 帅帅想请你帮忙写一个程序，对于任意矩阵，可以求出取数后的最大得分。由于得分可能会非常大，所以把值对1000000007取模
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param matrix int整型二维数组
# @return int整型
#
#由于各行的数据选择毫不相关且总得分为每行得分之和,可以先对每一行的数据独立计算最大得分
class Solution:
    def matrixScore(self, matrix: List[List[int]]) -> int:
        # write code here
        ret = 0
        for i in range(len(matrix)):
            ret = (ret % 1000000007 + self.help(matrix[i]) % 1000000007) % 1000000007
        return ret

    def help(self, aList):
        dp = [[0] * len(aList) for i in range(len(aList))]
        for i in range(len(aList)):
            dp[i][i] = 2 * aList[i]
        for i in range(len(aList) - 2, -1, -1):
            for j in range(i + 1, len(aList)):
                dp[i][j] = max(
                    2 * aList[i] + 2 * dp[i + 1][j], 2 * aList[j] + 2 * dp[i][j - 1]
                )
        #注:由于max操作对取余不保序，所以不可在计算dp过程取余
        return dp[0][-1]
################################################
# 给定一个长度为 n 的整数数组，和一个目标值 k ，请你找出这个整数数组中和大于等于 k 的最短子数组的长度。如果不存在和大于等于 k 的子数组则输出 -1。
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param nums int整型一维数组 
# @param k int整型 
# @return int整型
#
#双指针
class Solution:
    def shortestSubarray(self , nums: List[int], k: int) -> int:
        # write code here
        ret = len(nums)+1
        left, right, s = 0, 0, 0    
        while right < len(nums):
            while right < len(nums) and s < k:
                s += nums[right]
                right += 1
            while s >= k:
                s -= nums[left]
                left += 1
            ret = min(ret, right - left + 1)
        return ret if ret <= len(nums) else -1
################################################
# 给定两个长度为 n 和 m 的升序数组（后一个数一定大于等于前一个数），请你找到这两个数组中全部元素的中位数。
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param nums1 int整型一维数组 
# @param nums2 int整型一维数组 
# @return double浮点型
#
class Solution:
    def Median(self , nums1: List[int], nums2: List[int]) -> float:
        # write code here
        n = len(nums1) + len(nums2)
        return (self.help(nums1, 0, nums2, 0, (n-1)//2+1) + self.help(nums1, 0, nums2, 0, n//2+1)) / 2
    def help(self, nums1, i, nums2, j, k):
        #在两个有序数组nums[i:], nums2[j:]中找到第k个元素（例如找第一个元素，k=1，即nums[0]）
        if i >= len(nums1): return nums2[j+k-1]
        if j >= len(nums2): return nums1[i+k-1]
        if k == 1: return min(nums1[i], nums2[j])    #递归终止条件
        idx1, idx2 = i+k//2-1, j+k//2-1
        # 这两个数组的第K/2小的数字，若不足k/2个数字则赋值整型最大值，以便淘汰另一数组的前k/2个数字
        n1, n2 = nums1[idx1] if idx1 < len(nums1) else float('inf'), nums2[idx2] if idx2 < len(nums2) else float('inf')
        if n1 < n2: return self.help(nums1, idx1+1, nums2, j, k-k//2)    #二分
        else: return self.help(nums1, i, nums2, idx2+1, k-k//2)
################################################
# 已知矩阵的大小定义为矩阵中所有元素的和。
# 给定一个大小为N*N的矩阵，你的任务是找到最大的非空(大小至少是1 * 1)子矩阵。 比如，如下4 * 4的矩阵
# 0 -2 -7 0
# 9 2 -6 2
# -4 1 -4 1
# -1 8 0 -2 的最大子矩阵是
# 9 2
# -4 1
# -1 8 这个子矩阵的大小是15。
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param matrix int整型二维数组 
# @return int整型
#
#把二维数组最大子矩阵和 转换成 一维数组的最大子数组：
#把二维数组M x N 每一行分别相加，就可以得出一个一维数组(长度为N），
#这个一维数组的最大子数组和就是原矩阵中行数为M的一个最大子矩阵和，
#因此对子矩阵的行指标的起止位置枚举就可以得出最后结果
#时间复杂度为O(n^3),空间复杂度为O(n)
class Solution:
    def getMaxMatrix(self , matrix: List[List[int]]) -> int:
        # write code here
        ret = float('-inf')
        for s in range(len(matrix)):
            row_sum = matrix[s]
            ret = max(ret, self.help(row_sum))
            for e in range(s+1, len(matrix)):
                for j in range(len(matrix[0])):
                    row_sum[j] += matrix[e][j]
                ret = max(ret, self.help(row_sum))
        return int(ret)
    def help(self, aList):
        r, s = 0, 0
        ret = float('-inf')
        while r < len(aList):
            s += aList[r]
            ret = max(s, ret)
            r += 1
            if s <= 0:
                s = 0
        return ret
################################################
# 给定一个由 '[' ，']'，'('，‘)’ 组成的字符串，请问最少插入多少个括号就能使这个字符串的所有括号左右配对。
# 例如当前串是 "([[])"，那么插入一个']'即可满足。
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param s string字符串 
# @return int整型
#
#dp[i][j]为s[i...j]至少需要插入的括号数
#对s[i]和s[j]是否属于同一部分分类讨论:
#若属于同一部分，形为'(xxx)'，则s[i]s[j]必然为'()'或'[]'，此时dp[i][j] = dp[i+1][j-1]
#否则可将s[i]，s[j]分别划分到左右两部分，形为'(xxx)(yyy)'，对划分点分类讨论，dp[i][j] = dp[i][k]+dp[k+1][j], k=i,...,j-1
class Solution:
    def match(self , s: str) -> int:
        # write code here
        mp = {'(': ')', '[': ']'}
        dp = [[len(s)] * len(s) for i in range(len(s))]
        for i in range(len(s)):
            dp[i][i] = 1
            for j in range(i):
                dp[i][j] = 0        #当i>j时s[i...j]为空，无需插入
        for i in range(len(s)-2, -1, -1):
            for j in range(i+1, len(s)):
                if s[i] in mp and mp[s[i]] == s[j]: dp[i][j] = min(dp[i][j], dp[i+1][j-1])
                for k in range(i, j):
                    dp[i][j] = min(dp[i][j], dp[i][k]+dp[k+1][j])
        return dp[0][-1]
################################################
# 给定一个长度为 n 的字符串 s ，请你找出 s 的最长子串，这个子串满足所有字符都出现大于等于 k 次。请你返回这个子串的长度。
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param s string字符串 
# @param k int整型 
# @return int整型
#
#统计字符串 s 中每个字符的出现次数
#以每个出现次数小于 k 的字符作为分割点将s分为多段；如果不存在出现次数小于 k 的字符，则整个字符串 s 都满足条件，直接返回 s 的长度（递归终止条件）
#对分割后的每个子串，递归调用本函数，返回子串的最长子串长度。
#返回所有子串中最长的子串长度。
from collections import defaultdict
class Solution:
    def longestSubstring(self , s: str, k: int) -> int:
        # write code here
        ret = 0
        cnt = defaultdict(int)
        for c in s:
            cnt[c] += 1
        flag = False    #判断是否有分割点
        segs = [-1]     #虚拟分割点，方便后续遍历逻辑
        for i in range(len(s)):
            if cnt[s[i]] < k:
                segs.append(i)
                if not flag: flag = True
        segs.append(len(s))     #虚拟分割点，原因同上
        if not flag: return len(s)  #无分割点，直接返回字符串长度
        for i in range(len(segs)-1):
            ret = max(ret, self.longestSubstring(s[segs[i]+1:segs[i+1]], k))
        return ret
################################################
# 在河上有一座独木桥，一只青蛙想沿着独木桥从河的一侧跳到另一侧。在桥上有一些石子，青蛙很讨厌踩在这些石子上。
# 由于桥的长度和青蛙一次跳过的距离都是正整数，我们可以把独木桥上青蛙可能到达的点看成数轴上的一串整点：0，1，……，L（其中L是桥的长度）。
# 坐标为0的点表示桥的起点，坐标为L的点表示桥的终点。青蛙从桥的起点开始，不停的向终点方向跳跃。一次跳跃的距离是S到T之间的任意正整数（包括S,T）。
# 当青蛙跳到或跳过坐标为L的点时，就算青蛙已经跳出了独木桥。
# 题目给出独木桥的长度L，青蛙跳跃的距离范围S,T，桥上石子的位置。你的任务是确定青蛙要想过河，最少需要踩到的石子数。
# 其中正整数 l ，表示独木桥的长度。s，t，分别表示青蛙一次跳跃的最小距离，最大距离，数组 nums 中 m 个不同的正整数分别表示这 m 个石子在数轴上的位置（数据保证桥的起点和终点处没有石子）。
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param l int整型
# @param s int整型
# @param t int整型
# @param nums int整型一维数组
# @return int整型
#
class Solution:
    def crossRiver(self, l: int, s: int, t: int, nums: List[int]) -> int:
        # write code here
        def gcd(x, y):
            if y == 0:
                return x
            return gcd(y, x % y)

        def lcm(x, y):
            return x * y // gcd(x, y)

        k = lcm(s, t)
        if s == t:
            ans = 0
            for n in nums:
                if n % k == 0:
                    ans += 1
            return ans
        import sys

        nums.insert(0, 0)
        nums.append(l)
        nums.sort()

        b = []
        for i in range(1, len(nums)):
            diff = nums[i] - nums[i - 1]
            if diff >= 2 * k:
                b.append(diff % k + k)
            else:
                b.append(diff)
        flags = [0] * 20000
        for i in range(1, len(nums)):
            nums[i] = nums[i - 1] + b[i - 1]
            flags[nums[i]] = 1
        flags[nums[-1]] = 0
        dp = [sys.maxsize] * 20000
        dp[0] = 0
        for i in range(1, nums[-1] + t + 2):
            for j in range(s, t + 1):
                if i - j >= 0:
                    dp[i] = min(dp[i], dp[i - j] + flags[i])
        return dp[nums[-1]]
################################################
# 给定一个长度为 n 的正整数数组请你选出一个区间，使得该区间是所有区间中经过下述计算方法得到的最大值。
# 计算方法：区间最小值 * 区间和
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param a int整型一维数组 
# @return int整型
#
#单调栈求左边最近的小于a[i]和右边最近的小于a[i]的索引；前缀和求区间和
class Solution:
    def mintimessum(self , a: List[int]) -> int:
        # write code here
        monoStk = []
        left, right = [-1] * len(a), [len(a)] * len(a)
        preSum = [0] * len(a)
        for i in range(len(a)):
            if i == 0: preSum[i] = a[i]
            else: preSum[i] = a[i] + preSum[i-1]
            while monoStk and a[monoStk[-1]] >= a[i]:
                monoStk.pop()
            if monoStk: left[i] = monoStk[-1]
            monoStk.append(i)
        monoStk = []
        for i in range(len(a)-1, -1, -1):
            while monoStk and a[monoStk[-1]] >= a[i]:
                monoStk.pop()
            if monoStk: right[i] = monoStk[-1]
            monoStk.append(i)
        ret = 0
        for i in range(len(a)):     #注意特殊情况讨论
            if left[i] == -1 and right[i] == len(a):    #a[i]为数组最小值
                ret = max(ret, a[i] * preSum[-1])
            elif left[i] == -1:                         #a[i]左边全比它大
                ret = max(ret, a[i] * preSum[right[i]-1])
            elif right[i] == len(a):                    #a[i]右边全比它大
                ret = max(ret, a[i] * (preSum[-1]-preSum[left[i]]))
            else: ret = max(ret, a[i] * (preSum[right[i]-1]-preSum[left[i]]))
        return ret
################################################
# 给定一个字符串形式的表达式 s ，请你实现一个计算器并返回结果。字符串中包含 + , -  , ( , ) ，保证表达式合法
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param s string字符串 
# @return int整型
#将字符串预处理为统一格式'+/- number1 +/- number2', 括号可视作number的一部分
#结果即为每个+/- number相加即可, number可由递归调用得到 
class Solution:
    def calculate(self , s: str) -> int:
        # write code here
        if not s: return 0  #空字符可看作是s+'+0', 直接返回0
        digits = '0123456789'
        num = 0
        if s[0] not in  '+-': s = '+' + s   #预处理
        sign = '+'
        if s[0] == '-': sign = '-'  #记录当前num符号
        i = 1                       #记录number结束位置
        if s[1] in digits:          #若number只由数字组成, 如123, 直接提取
            while  i < len(s) and s[i] in digits:
                num = num * 10 + (ord(s[i])-ord('0'))
                i += 1
        if s[1] == '(':     #若number为括号内式子的结果, 可利用递归求取式子结果
            stk = []
            while i < len(s):
                if s[i] == '(': stk.append('(')
                if s[i] == ')': stk.pop()
                i += 1
                if not stk: break
            num = self.calculate(s[2:i-1])
        #结果即为剩余字符串结果与当前结果之和
        if sign == '-': return -num + self.calculate(s[i:])
        else: return num + self.calculate(s[i:])
################################################
# 给定一个仅包含 0 和 1 ，大小为 n*m 的二维二进制矩阵，找出仅包含 1 的最大矩形并返回面积。
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param matrix int整型二维数组 
# @return int整型
#
#利用动态规划求出以matrix[i][j]结尾的第j列的连续的1的个数, 将该问题转为n个直方图最大矩形面积
#即对于每一行构造直方图heights, heigts[j]即为之前求出的连续1的个数
class Solution:
    def maximalRectangle(self , matrix: List[List[int]]) -> int:
        # write code here
        ret = 0
        heights = [0] * len(matrix[0])
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j]: heights[j] += 1
                else: heights[j] = 0
            ret = max(ret, self.help(heights))
        return ret
    def help(self, heights):
        #利用单调栈求解直方图最大矩形面积
        monoStk = []
        ret = 0
        left, right = [-1] * len(heights), [len(heights)] * len(heights)
        for i in range(len(heights)):
            while monoStk and heights[i] <= heights[monoStk[-1]]:
                monoStk.pop()
            if monoStk: left[i] = monoStk[-1]
            monoStk.append(i)
        monoStk = []
        for i in range(len(heights)-1, -1, -1):
            while monoStk and heights[i] <= heights[monoStk[-1]]:
                monoStk.pop()
            if monoStk: right[i] = monoStk[-1]
            monoStk.append(i)
            ret = max(ret, heights[i] * (right[i] - left[i] - 1))
        return ret
################################################
# 给定正整数 n 和 k ，请你找出 [1,n] 内的字典序第 k 小的数。
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param n int整型 
# @param k int整型 
# @return int整型
#利用字典树的思想
#在[1, n]构成的字典树中，先序遍历可得到[1, n]的字典序升序，层次遍历可得到数字升序，利用这点可以对查找过程做优化
class Solution:
    def findKth(self , n: int, k: int) -> int:
        # write code here
        cur = 1
        while k > 1:
            cnt = self.help(cur, n)
            if cnt < k:
                cur += 1
                k -= cnt
            else:
                k -= 1
                cur *= 10
        return cur
    def help(self, root, n):    #计算[1, n]字典树中以root为根节点的子树所含节点数
        left, right = root, root
        ret = 0
        while left <= n:
            ret += min(right, n) - left + 1
            left = left * 10
            right = right * 10 + 9
        return ret
################################################
# 给出4个1-10的数字，通过加减乘除运算，得到数字为24就算胜利,除法指实数除法运算,运算符仅允许出现在两个数字之间
# 本题对数字选取顺序无要求，但每个数字仅允许使用一次，且需考虑括号运算
# 此题允许数字重复，如3 3 4 4为合法输入，此输入一共有两个3，但是每个数字只允许使用一次，则运算过程中两个3都被选取并进行对应的计算操作。
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param nums int整型一维数组 
# @return bool布尔型
#
class Solution:
    def Point24(self , nums: List[int]) -> bool:
        # write code here
        if len(nums) == 1: return nums[0] == 24
        nums.sort(reverse=True)    #排序，仅考虑num1 >= num2，避免重复讨论
        for i in range(len(nums)-1):
            for j in range(i+1, len(nums)):
                num2, num1 = nums.pop(j), nums.pop(i)
                nums.append(num1+num2)
                if self.Point24(nums): return True
                nums.remove(num1+num2)
                nums.append(num1-num2)
                if self.Point24(nums): return True
                nums.remove(num1-num2)
                nums.append(num1*num2)
                if self.Point24(nums): return True
                nums.remove(num1*num2)
                if num2:
                    nums.append(num1/num2)
                    if self.Point24(nums): return True
                    nums.remove(num1/num2)
                nums.insert(i, num1)    #回溯, 还原nums,注意num1和num2的先后插入顺序
                nums.insert(j, num2)
        return False
################################################
# 给定一个长度为 n 的正整数数组 nums 和一个目标整数 k ，返回数组中的 牛连续子数组 的数目。
# 如果 nums 中的某个连续子数组中不同的整数个数恰好是 k 个，则称这个连续子数组为 牛连续子数组，不同位置的连续子数组可能一样，都算入最终数目里。
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param nums int整型一维数组 
# @param k int整型 
# @return int整型
#
from collections import defaultdict
class Solution:
    def nowsubarray(self , nums: List[int], k: int) -> int:
        # write code here
        return self.help(nums, k) - self.help(nums, k-1)
    def help(self, nums, k):
        '''至多包含k个不同整数的子数组个数'''
        i, j, cnt = 0, 0, 0
        cnt = defaultdict(int)  #记录[i, j)区间各整数的出现次数
        ret = 0
        while j < len(nums):
            cnt[nums[j]] += 1
            j += 1
            while len(cnt) > k:
                cnt[nums[i]] -= 1
                if not cnt[nums[i]]: cnt.pop(nums[i])
                i += 1      #i: 以j为右端点时, 满足条件的最长子数组的左端点
            ret += j - i    #对于每个右端点j, 满足条件的子数组个数即为j-i
        return ret
################################################
# 给定一个长度为 n 的正整数数组 nums ，返回所有距离对中第 k 小的距离。
# 距离对定位为：nums[i] , nums[j] 的差值的绝对值称为距离 
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param nums int整型一维数组 
# @param k int整型 
# @return int整型
#
from bisect import bisect_left
class Solution:
    def kthDistance(self , nums: List[int], k: int) -> int:
        # write code here
        nums.sort()
        left, right = 0, nums[-1] - nums[0]
        while left < right:
            mid = (left + right) // 2
            if self.help(nums, mid) < k: left = mid + 1
            else: right = mid
        return left
    def help(self, nums, dist):     #计算距离不超过dist的数对(nums[i],nums[j])的个数
        ret = 0
        for j in range(len(nums)):  #固定nums[j], 二分查找nums[i]可取的最小值
            i = bisect_left(nums, nums[j]-dist)
            ret += j - i            #[i,j)内的数与nums[j]组成数对都可满足条件
        return ret
################################################
# 在Mars星球上，每个Mars人都随身佩带着一串能量项链。在项链上有N颗能量珠。能量珠是一颗有头标记与尾标记的珠子，这些标记对应着某个正整数。
# 并且，对于相邻的两颗珠子，前一颗珠子的尾标记一定等于后一颗珠子的头标记。因为只有这样，通过吸盘（吸盘是Mars人吸收能量的一种器官）的作用，这两颗珠子才能聚合成一颗珠子，同时释放出可以被吸盘吸收的能量。
# 如果前一颗能量珠的头标记为m，尾标记为r，后一颗能量珠的头标记为 r，尾标记为 n，则聚合后释放的能量为 （Mars单位），新产生的珠子的头标记为 m，尾标记为 n。
# 需要时，Mars人就用吸盘夹住相邻的两颗珠子，通过聚合得到能量，直到项链上只剩下一颗珠子为止。显然，不同的聚合顺序得到的总能量是不同的，请你设计一个聚合顺序，使一串项链释放出的总能量最大。
# 例如：设N=4，4颗珠子的头标记与尾标记依次为(2，3) (3，5) (5，10) (10，2)。我们用记号⊕表示两颗珠子的聚合操作，(j⊕k)表示第j，k两颗珠子聚合后所释放的能量。则第4、1两颗珠子聚合后释放的能量为：(4⊕1)=10*2*3=60。
# 这一串项链可以得到最优值的一个聚合顺序所释放的总能量为
# ((4⊕1)⊕2)⊕3）=10*2*3+10*3*5+10*5*10=710。
# 第四颗珠子先和第一颗珠子聚合得到一个 (10,3) 的珠子，然后与第二颗珠子聚合得到 (10,5) ，然后与第三颗珠子聚合得到 (10,10)
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param n int整型 
# @param num int整型一维数组 
# @return int整型
#
class Solution:
    def necklace(self , n: int, num: List[int]) -> int:
        # write code here
        ret = 0
        num[n:n] = num[:]
        dp = [[0] * (2*n-1) for i in range(2*n-1)]
        for i in range(2*n-2, -1, -1):
            for j in range(i+1, min(i+n, 2*n-1)):
                for k in range(i, j):
                    dp[i][j] = max(dp[i][j],
                                    dp[i][k]+dp[k+1][j]+num[i]*num[k+1]*num[j+1])
        for i in range(n):
            ret = max(ret, dp[i][i+n-1])
        return ret % 1000000007
################################################
# 小红拿到了一个数组，她希望取其中的三个数，使得以这三个数为边长的三角形周长尽可能小
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param nums int整型一维数组 
# @return int整型
#
#point1: 假设最短三角形的三条边依次是a、b、c(a<b<c), 则b、c在排序后的数组中一定是相邻的;
#point2: 将c设为遍历变量，便可以直接确定b，接下来只需要找到a就可以了，根据三角形性质，有#a>c-b，借助二分查找法我们可以查找到0~b范围内比c-b大的最小值;
#优化: 当已有最短周长<2c+1时可提前结束遍历，因为当我们确定c时，由它所构成的三角形的最短周长为: c+b+(c-b+1)=2c+1，这个值越往后遍历只会越来越大
import bisect
class Solution:
    def hongstriangle(self , nums: List[int]) -> int:
        # write code here
        nums.sort()
        k = 2
        ret = float('inf')
        while k < len(nums):
            if ret <= 2 * nums[k] + 1: break
            i = bisect.bisect_right(nums, nums[k]-nums[k-1], hi=k-1)
            if i == k-1:    #此时num[:k-1]中没有比nums[k]-nums[k-1]大的数，直接跳过
                k += 1
                continue
            else:
                ret = min(ret, nums[i]+nums[k-1]+nums[k])
                k += 1
        return ret
################################################
# 请实现一个函数用来判断字符串str是否表示数值（包括科学计数法的数字，小数和整数）。
# 科学计数法的数字(按顺序）可以分成以下几个部分:
# 1.若干空格
# 2.一个整数或者小数
# 3.（可选）一个 'e' 或 'E' ，后面跟着一个整数(可正可负)
# 4.若干空格

# 小数（按顺序）可以分成以下几个部分：
# 1.若干空格
# 2.（可选）一个符号字符（'+' 或 '-'）
# 3. 可能是以下描述格式之一:
# 3.1 至少一位数字，后面跟着一个点 '.'
# 3.2 至少一位数字，后面跟着一个点 '.' ，后面再跟着至少一位数字
# 3.3 一个点 '.' ，后面跟着至少一位数字
# 4.若干空格

# 整数（按顺序）可以分成以下几个部分：
# 1.若干空格
# 2.（可选）一个符号字符（'+' 或 '-')
# 3. 至少一位数字
# 4.若干空格

# 例如，字符串["+100","5e2","-123","3.1416","-1E-16"]都表示数值。
# 但是["12e","1a3.14","1.2.3","+-5","12e+4.3"]都不是数值。
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param str string字符串 
# @return bool布尔型
#
#正则表达式匹配法
import re
class Solution:
    def isNumeric(self , str: str) -> bool:
        # write code here
        match_Obj = re.match('^\s*[+-]{0,1}((\d)+((\.)(\d)+){0,1}|((\.)(\d)+)|((\d)+(\.)))([eE][+-]{0,1}[\d]+){0,1}\s*$',str)
        if match_Obj:
            return True
        else:
            return False
#遍历判断法            
class Solution:
    def isNumeric(self , str ):
        # write code here
        str = str.strip()       #去空格
        if 'e' not in str and 'E' not in str:       #若不为科学计数法，判断整/小
            return self.decimal(str) or self.integer(str)
        if str.count('e') == 1 and str.count('E') == 0:
            sList = str.split('e')      
            #若可能为科学计数，按'e'或'E'分割字符串，判断前后两部分是否满足条件
            return self.integer(sList[1]) and \
            (self.decimal(sList[0]) or self.integer(sList[0]))
        if str.count('e') == 0 and str.count('E') == 1:
            sList = str.split('E')
            return self.integer(sList[1]) and \
            (self.decimal(sList[0]) or self.integer(sList[0]))
        return False
    def decimal(self, str):     #判断是否为小数
        if not str: return False        #特判空字符直接返回False
        digits = '0123456789'
        i = 0
        if str[0] in '+-': i += 1       #判断是否包含'+-'
        #以'.'将str分割为两部分，判断两部分是否至少有一部分为数字
        flag1 = False
        while i < len(str) and str[i] in digits:
            i += 1
            flag1 = True
        if i == len(str) or str[i] != '.': return False
        else: i += 1
        flag2 = False
        while i < len(str) and str[i] in digits:
            i += 1
            flag2 = True
        return i == len(str) and (flag1 or flag2)
    def integer(self, str):     #判断除'+-'外是否全为数字
        if not str: return False
        digits = '0123456789'
        i = 0
        if str[0] in '+-': i += 1
        flag = False
        while i < len(str) and str[i] in digits:
            i += 1
            flag = True
        return flag and i == len(str)
################################################
# 给定一个字符串s，里面可能含有若干括号和小写英文字母，请你判断最长有效的括号字符子序列有哪些，放在一个数组里面返回(你不用在乎序列在数组里面的顺序)。
# 最长有效括号字符子序列的标准如下:
# 1.每一个左括号，必须有对应的右括号和它对应
# 2.字母的位置以及存在对括号字符子序列的合法性不受影响
# 3.子序列是不连续的，比如"()("的子序列有"()",")(","()(","(("
# 4.相同的括号字符子序列只保留一个，比如"())"有2个子序列"()"，但是最后只保留一个"()"
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param s string字符串 
# @return string字符串一维数组
#
class Solution:
    def maxValidParenthesesStr(self , s: str) -> List[str]:
        # write code here
        self.maxLen = 0     #全局变量，维护当前最大有效序列长度
        mp = {'(': 1, ')': -1}
        for i in range(26):
            mp[chr(97+i)] = 0      #建立映射关系，通过序列之和判断是否可能合法
        ret = []
        ret.append('')
        used = [0] * len(s)     #记录当前所用字符，避免重复讨论
        self.dfs(s, 0, '', 0, used, mp, ret)
        return ret
    def dfs(self, s, idx, tmp, tmpSum, used, mp, ret):
        '''tmp: 当前序列, idx: 下一个可选字符的起始索引'''
        if len(s) - idx + len(tmp) < self.maxLen: return    #当最长长度不可能增加时停止
        for i in range(idx, len(s)):
            #相邻两字符相同时，仅取前和仅取后情况相同，无需重复讨论
            if i > 0 and not used[i-1] and s[i] == s[i-1]: continue
            #当tmp+s[i]的右括号比左括号多时，不可能为合法序列
            if tmpSum + mp[s[i]] < 0: continue
            used[i] = 1
            if tmpSum + mp[s[i]] == 0:  #此时为合法序列
                if self.maxLen < len(tmp)+1:    #判断当前序列长度和当前最长长度大小
                    self.maxLen = len(tmp) + 1
                    #important: 此处ret必须为原地操作, 不可ret=[], 会改变ret引用原对象
                    ret.clear()
                    ret.append(tmp+s[i])
                elif self.maxLen == len(tmp)+1:
                    ret.append(tmp+s[i])
                self.dfs(s, i+1, tmp+s[i], tmpSum+mp[s[i]], used, mp, ret)
            else:
                self.dfs(s, i+1, tmp+s[i], tmpSum+mp[s[i]], used, mp, ret)
            used[i] = 0
################################################
# 给定一个长度为 n 的正整数数组，和一个窗口长度 k ，有一个长度为 k 的窗口从最左端滑到最右端。请你算出所有窗口的中位数。
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param nums int整型一维数组 
# @param k int整型 
# @return double浮点型一维数组
#
import heapq
from collections import defaultdict
class Solution:
    def slidewindow(self , nums: List[int], k: int) -> List[float]:
        # write code here
        self.small, self.large, self.delCnt = [], [], defaultdict(int)
        self.sCnt, self.lCnt = 0, 0
        heapq.heapify(self.small); heapq.heapify(self.large)
        l, r = 0, 0
        ret = []
        if k % 2:
            while r < len(nums):               
                self.insert(nums[r])
                r += 1
                k -= 1
                if k < 0:
                    self.remove(nums[l])
                    l += 1
                    k += 1
                if k == 0:
                    #print(self.small, self.large)
                    ret.append(self.large[0])                 
        else:
            while r < len(nums):               
                self.insert(nums[r])
                r += 1
                k -= 1
                if k < 0:
                    self.remove(nums[l])
                    l += 1
                    k += 1
                if k == 0:
                    #print(self.small, self.large)
                    ret.append(0.5*(self.large[0]-self.small[0])) 
        return ret
    def insert(self, num):
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))
        self.lCnt += 1
        if  self.lCnt > self.sCnt+1:
            tmp = heapq.heappop(self.large)
            self.lCnt -= 1
            heapq.heappush(self.small, -tmp)
            self.sCnt += 1
        self.balance()
    def remove(self, num):
        self.delCnt[num] += 1
        if num < self.large[0]: self.sCnt -= 1
        else: self.lCnt -= 1
        self.balance()
    def prune(self):
        while self.large and self.delCnt[self.large[0]]:
            self.delCnt[self.large[0]] -= 1
            heapq.heappop(self.large)
        while self.small and self.delCnt[-self.small[0]]:
            self.delCnt[-self.small[0]] -= 1
            heapq.heappop(self.small)            
    def balance(self):
        if self.lCnt < self.sCnt:
            heapq.heappush(self.large,-heapq.heappop(self.small))
            self.lCnt += 1; self.sCnt -= 1
        elif self.lCnt > self.sCnt+1:
            heapq.heappush(self.small, -heapq.heappop(self.large))
            self.lCnt -= 1; self.sCnt += 1
        self.prune()
################################################
# 给定一个字符串，你可以在其任何位置插入新字符，请你算出要把这个字符串变成回文字符串最少需要几次插入操作。
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param str string字符串 
# @return int整型
#
#若s[i]==s[j], dp[i][j] = dp[i+1][j-1]; 
#若s[i]!=s[j], dp[i][j] = min(dp[i+1][j], dp[i][j-1])+1, 即对s[i]或s[j]为最外层字符讨论
class Solution:
    def minInsert(self , str: str) -> int:
        # write code here
        dp = [[0] * len(str) for i in range(len(str))]
        for i in range(len(str)-2, -1, -1):
            for j in range(i+1, len(str)):
                if str[i] == str[j]:
                    dp[i][j] = dp[i+1][j-1]
                else:
                    dp[i][j] = min(dp[i+1][j], dp[i][j-1]) + 1
        return dp[0][-1]
################################################
# 给定一个长度为 n 的数组 nums ，请你返回一个新数组 count ，其中 count[i] 是 nums[i] 右侧小于 nums[i] 的元素个数。
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param nums int整型一维数组 
# @return int整型一维数组
#
import bisect
class Solution:
    def smallerCount(self , nums: List[int]) -> List[int]:
        # write code here
        tmp, ret = [], []   #tmp维护当前数右侧的数的升序序列
        for i in range(len(nums)-1, -1, -1):
            idx = bisect.bisect_left(tmp, nums[i])
            ret.append(idx)
            tmp.insert(idx, nums[i])
        ret.reverse()
        return ret
################################################
# 设一个n个节点的二叉树tree的中序遍历为（l,2,3,…,n），其中数字1,2,3,…,n为节点编号。
# 每个节点都有一个分数（均为正整数），记第j个节点的分数为di，tree及它的每个子树都有一个加分，任一棵子树subtree（也包含tree本身）的加分计算方法如下：
# subtree的左子树的加分× subtree的右子树的加分＋subtree的根的分数
# 若某个子树为主，规定其加分为1，叶子的加分就是叶节点本身的分数。不考虑它的空子树。 试求一棵符合中序遍历为（1,2,3,…,n）且加分最高的二叉树tree。
# 要求输出：1）tree的最高加分；2）tree的前序遍历
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param scores int整型一维数组 
# @return int整型二维数组
#
class Solution:
    def scoreTree(self , scores: List[int]) -> List[List[int]]:
        # write code here
        dp = [[0] * 35 for i in range(35)]  #dp[i][j]为区间[i, j]的最高加分
        self.f = [[0] * 35 for i in range(35)]  #f[i][j]为最高加分对应的根节点
        self.seq = []
        ret = []
        for lenth in range(len(scores)):
            for i in range(len(scores)-lenth):
                if lenth == 0:
                    dp[i][i] = scores[i]
                    self.f[i][i] = i
                    continue
                j = i + lenth
                for k in range(i, j+1):
                    left = 1 if k == i else dp[i][k-1]
                    right = 1 if k == j else dp[k+1][j]
                    s = left * right + scores[k]
                    if dp[i][j] < s:
                        dp[i][j] = s
                        self.f[i][j] = k
        ret.append([dp[0][len(scores)-1]])
        self.find(0, len(scores)-1)
        ret.append(self.seq)
        return ret
    def find(self, l, r):
        if l <= r:
            self.seq.append(self.f[l][r]+1)
            self.find(l, self.f[l][r]-1)
            self.find(self.f[l][r]+1, r)
################################################
# Git 是一个常用的分布式代码管理工具，Git 通过树的形式记录文件的更改历史（例如示例图），树上的每个节点表示一个版本分支，工程师经常需要找到两个分支的最近的分割点。
# 例如示例图中 3,4 版本的分割点是 1。3,5 版本的分割点是 0。
# 给定一个用邻接矩阵 matrix 表示的树，请你找到版本 versionA 和 versionB 最近的分割点并返回编号。
# 注意：
# 1.矩阵中从第一行 （视为节点 0 ）开始，表示与其他每个点的连接情况，例如 [01011,10100,01000,10000,10000] 表示节点 0 与节点 1 ， 3 ， 4相连，节点 1 与节点 0 ， 2相连，其他点的以此类推。
# 2.并不保证是一棵二叉树，即一个节点有可能有多个后继节点，我们把节点 0 视为树的根节点。
class Solution:
    def Git(self , matrix: List[str], versionA: int, versionB: int) -> int:
        # write code here     
        global alist,blist
        alist=[]#用来保存a的路径
        blist=[]#用来保存b的路径
        tree={}
        for i in range(len(matrix)):#整理为字典，树形结构
            tree[i]=[]
            relation=list(matrix[i])
            index=0
            while len(relation)>0:
                r=relation.pop(0)
                if r=="1":
                    tree[i].append(index)
                index+=1      
        def walk(tmplist,pre,node):
            tmplist.append(node)
            if node ==versionA:#找到a，保存路径
                global alist
                alist=tmplist.copy()
            if node == versionB:#找到b，保存路径，注意这里a和b可能是同一个值，所以不要用else
                global blist
                blist=tmplist.copy()
            for i in tree[node]:#遍历执行下级节点
                if i == pre:#注意跳过来时的节点
                    continue
                walk(tmplist,node,i)
                tmplist.pop()
        walk([],0,0)
        res=0
        while len(alist)>0 and len(blist)>0:#逐个弹出首位比较，直到不相同或者路径为空
            a=alist.pop(0)
            b=blist.pop(0)
            if a==b:
                res=a
            else:
                break
        return res
