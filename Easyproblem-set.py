#判断一个链表是否有环，如果采用暴力求解，时间复杂度为O(n^2)
#此题可考虑采用快慢指针，快指针fast每次前进两步，而慢指针slow前进一步
#若无环，那么fast或fast.next必然先到达链表末尾None;
#否则,slow进入环前,fast已进入环，当slow遍历环中所有节点时,fast与slow必然会相遇
#时间复杂度O(n),空间复杂度O(1),following is code:
###################################################
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
#
# @param head ListNode类 
# @return bool布尔型
#
class Solution:
    def hasCycle(self , head ):
        if head:
            slow = head
            fast = head
            flag = False
            while not flag and fast and fast.next:
                    slow = slow.next
                    fast = fast.next.next
                    if slow == fast:
                        flag = True
        else:
            flag = False
        return flag
###################################################
#Fibinacci数列非递归
#每次记录并更新两个值(当前值及其前一个,类似于动态规划),following is code:
###################################################
class Solution:
    def Fibonacci(self, n):
        zero,one=0,1
        for _ in range(n):
            zero,one=one,zero+one
        return zero
###################################################
#给定一个m x n大小的矩阵（m行，n列），按螺旋的顺序返回矩阵中的所有元素。
# @param matrix int整型二维数组 
# @return int整型一维数组
#可利用递归，重点在于判断矩阵行列的奇偶性
class Solution:
    def spiralOrder(self , matrix ):
        if not matrix:
            return []
        else:
            return spiralHelp(matrix,0,len(matrix)-1,len(matrix[0])-1,[])
def spiralHelp(matrix,start,m,n,li):
    if start <= n and start <= m:
        for col in range(start,n+1):
            li.append(matrix[start][col])
        for row in range(start+1,m+1):
            li.append(matrix[row][n])
        for col in range(n-1,start-1,-1):
            if start != m:                     #若无此判断,奇*奇阶矩阵不满足,即会重复遍历,下同
                li.append(matrix[m][col])
        for row in range(m-1,start,-1):
            if start != n:                     #若无此判断,奇*偶阶矩阵不满足
                li.append(matrix[row][start])
        spiralHelp(matrix,start+1,m-1,n-1,li)
    return li
###################################################
#假设你有一个数组，其中第 i 个元素是股票在第 i 天的价格。
#你有一次买入和卖出的机会。（只有买入了股票以后才能卖出）,计算可以获得的最大收益。 
# @param prices int整型一维数组 
# @return int整型
class Solution:
    def maxProfit(self , prices ):
        maxprofit = 0
        dp = 0                                       #dp记为第i天卖出所获最大收益,因为dp[i]只可能使用了上一时刻的dp[i-1],而且ans可以在dp生成过程中算出来,所以没必要把dp开成数组
        for i in range(1,len(prices)):
            dp = max(dp, 0) + prices[i]-prices[i-1]  #则dp[i+1]=max(prices[i+1]-prices[i]+dp[i],prices[i+1]-prices[i])
            if dp > maxprofit:
                maxprofit = dp
        return maxprofit
###################################################    
#计算给定n个数，在它们中间添加"+"， "*"， "("， ")"符号，能够获得的最大值。
#动态规划：对数字进行分组，例如n==4时，依次计算两个数字一组，三个数字一组，四个数字一组的最大值
#注意到计算三个数字一组的情况用到了两个数字一组的最大值，据此可以考虑用动态规划并写出转移方程
#
while True:
    try:
        nums = list(map(int, input().split()))
        N = len(nums)
        dp = [[nums[i] if i==j else 0 for j in range(N)] for i in range(N)] # dp[i][j] 第i个数到第j个数的最大
        for r in range(2,N+1):          #对数字进行分组
            for i in range(N-r+1):
                j = i+r-1
                for k in range(i,j):        #在第k个数后断开
                    dp[i][j] = max(dp[i][j],dp[i][k]+dp[k+1][j],dp[i][k]*dp[k+1][j])
        print(dp[0][N-1])
    except:
        break
###################################################    
#输入一个字符串(只包含a-z)，在a-z中任选一个插入，一共可以形成多少种不同的字符串
while True:
    try:
        s = input()
        # 可在 (len(s) + 1)个位置插入26字符中的一个
        # 当插入相同字符时，放在选定字符前后只算一个
        # 以a为例，在a的旁边，每个a，有2种方法（左边插、右边插），
        # 但实际只算1种，所以对于a，要减去a出现的次数（无论所有的a是否相邻！）；
        # 以此类推，减去所有字符的出现次数，那就是减去字符串长度了。
        res = (len(s)+1)*26-len(s)
        print(res)
    except:
        break
###################################################
#给定一个二叉树，返回该二叉树层序遍历的结果
#如输入{1,2,3,4,#,#,5}，输出[[1],[2,3],[4,5]]，其中#代表None
#
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# 
# @param root TreeNode类 
# @return int整型二维数组
class Solution:
    def levelOrder(self , root ):
        if root:
            return bfs(root,[])
        else:
            return []
def bfs(root,bfslist):
    queue = []
    queue.append(root)
    bfslist.append([root.val])
    length = 1
    while queue:
        children = []
        for i in range(length):            #遍历每一层节点
            parent = queue.pop(0)
            if parent.left:
                children = children + [parent.left.val]
                queue.append(parent.left)
            if parent.right:
                children = children + [parent.right.val]
                queue.append(parent.right)
        if children:
            bfslist.append(children)
        length = len(queue)            #区分：标识当前层和下一层的分界位置
    return bfslist
###################################################
#给定一个数组arr，返回arr的最长无的重复子串的长度(无重复指的是所有数字都不相同)。
# 
# @param arr int整型一维数组 the array
# @return int整型
#
class Solution:
    def maxLength(self , arr ):
        if len(arr) == 0:
            return 0
        d = {}
        startpos = 0
        endpos = 1
        d[arr[startpos]] = startpos     #d记录从startpos开始扫描过的元素，值：索引
        ans = 1
        while endpos <= len(arr) - 1:
            if arr[endpos] not in d:
                d[arr[endpos]] = endpos
                ans = max(ans,endpos-startpos+1)     #更新当前最大长度
            else:
                startpos = d[arr[endpos]]+1         #有重复元素时，更新startpos为重复元素后一个位置,此处可考虑用取大函数简化代码
                for i in list(d.keys()):            #注意不能使用for i in d（由于字典在变化，d.keys()在变化），所以先将d.keys()提取出来再遍历
                    if i != arr[endpos]:            #从d中startpos删除至与arr[endpos]等值元素
                        del d[i]
                    else:
                        break
            endpos += 1
        return ans
###################################################
#数组中只出现一次的数（其它数出现k次）
#时间复杂度 O(32n)，空间复杂度 O(1)
# @param arr int一维数组 
# @param k int 
# @return int
##把每个数的二进制表示的每一位加起来，对于每一位的和，如果能被k整除，那对应那个只出现一次的数字的那一位就是0，否则对应的那一位是1
class Solution:
    def foundOnceNumber(self , arr , k ):
        positive = 0
        negative = 0
        for num in arr:
            if num < 0: negative += 1
            if num > 0: positive += 1         #统计正负个数,便于后面使用位运算
        binList = [0] * 32
        res = 0
        for pos in range(32):
            for num in arr:
                binList[pos] += (abs(num) >> pos & 1)      #IMPORTANT: 1.先取绝对值再移位 2.与1作按位与&可得二进制末位
            binList[pos] %= k
            res += binList[pos] << pos
        if negative % k: return -res          #判断正负号
        return res
###################################################
#给定一个无序数组 arr , 其中元素可正、可负、可0。给定一个整数 k
#求 arr 所有连续子数组中累加和为k的最长连续子数组长度。保证至少存在一个合法的连续子数组。
#
# max length of the subarray sum = k
# @param arr int整型一维数组 the array
# @param k int整型 target
# @return int整型
#
class Solution:
    def maxlenEqualK(self , arr , k ):
        tmpDict = {0: -1}    #记录出现过的tmpSum值和的它出现的最早位置，即第一次出现的位置。key代表出现过的tmpSum,value代表第一次出现的位置
        tmpSum = 0           #从arr[0]一直加到arr[i]的累加和
        maxlen = 0
        for i in range(len(arr)):
            tmpSum += arr[i]
            #IMPORTANT: 如果tmpSum-k存在，由于map的value记录的是最早出现的位置，所以此时arr[j+1…i]是以arr[i]结尾的所有子数组中，累加和为k的子数组中最长的
            if tmpSum - k in tmpDict : maxlen = max(maxlen, i - tmpDict[tmpSum - k])
            if tmpSum not in tmpDict: tmpDict[tmpSum] = i
        return maxlen
###################################################
#给定一个长度为 n 的非降序数组和一个非负数整数 k ，要求统计 k 在数组中出现的次数
# @param data int整型一维数组 
# @param k int整型 
# @return int整型
#
class Solution:
    def GetNumberOfK(self , data: List[int], k: int) -> int:
        if not data: return 0
        leftmark = findMin(data, k)
        rightmark = findMax(data, k)
        return rightmark - leftmark
def findMin(arr, k):
    '''寻找第一个 ≥k 的位置'''
    left = 0
    right = len(arr)      #相当于添加一个无穷大在data末尾,下同
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < k: left = mid + 1 
        else: right = mid
    return left
def findMax(arr, k):
    '''寻找第一个 >k 的位置'''
    left = 0
    right = len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] <= k: left = mid + 1
        if arr[mid] > k: right = mid
    return left
        # write code here
