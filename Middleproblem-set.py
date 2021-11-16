#kmp算法应用：给你一个文本串 T ，一个非空模板串 S ，问 S 在 T 中出现了多少次
#空间复杂度 O(len(S))，时间复杂度 O(len(S)+len(T))
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
# 计算模板串S在文本串T中出现了多少次
# @param S string字符串 模板串
# @param T string字符串 文本串
# @return int整型
#
class Solution:
    def kmp(self , S , T ):
        i, j = 0, 0
        lenthS, lenthT = len(S), len(T)
        if lenthS > lenthT: return 0
        cnt = 0
        nextArr = getNext(S)
        while lenthT - i >= lenthS - j:
            if j != -1 and T[i] == S[j]:
                i += 1
                j += 1
            elif j != -1 and T[i] != S[j]: j = nextArr[j]
            else: 
                i += 1
                j += 1
            if j == lenthS:      #匹配到一个模板串后,i,j更新规则如下:
                i -= 1           #i应为当前匹配子串的最后一个字符
                j = nextArr[-1]  #j应为nextArr对应最后一个字符的值
                cnt += 1
        return cnt
def getNext(S):
    nextList = [-1, 0]
    for i in range(2, len(S)):
        tmp = nextList[i-1]
        while tmp != -1:
            if S[i-1] == S[tmp]: break
            tmp = nextList[tmp]
        nextList.append(tmp + 1)
    return nextList
        # write code here
###################################################
#对于一个给定的链表，返回环的入口节点，如果没有环，返回None
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
#
# @param head ListNode类 
# @return ListNode类
###################################################
class Solution:
    def detectCycle(self , head ):
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if fast == slow:          #利用快慢指针判断是否有环并找相遇点
                break
        if fast and fast.next:        #找到相遇点后，将slow指向头节点
            slow = head
            while fast != slow:       #slow与fast以相同速度移动，直到相遇，相遇点即为入口节点
                slow = slow.next
                fast = fast.next
            return slow
        return None
        # write code here
###################################################
#给定两个字符串str1和str2,输出两个字符串的最长公共子串
# longest common substring
# @param str1 string字符串 the string
# @param str2 string字符串 the string
# @return string字符串
#遍历其中一个字符串，maxlen记录当前已找到公共子串的最大长度，找长度为maxlen+1的第一个公共子串，找到后，更新maxlen，ans
#奥义：每次更新maxlen和ans后，长度更长(即maxlen+1)的第一个公共子串只会以当前字符之后的字符为结尾，即按照我们的更新规则，当前字符前不会存在长度更长的公共子串
#因此复杂度为O(len(str1))
class Solution:
    def LCS(self , str1 , str2 ):
        if len(str1) < len(str2):
            str1,str2 = str2,str1
        maxlen = 0
        ans = ''
        for i in range(len(str1)):
            if str1[i - maxlen:i + 1] in str2: #for遍历比in判断的复杂度可能低一些,因此在开始先判断字符串长度大小关系进行优化
                ans = str1[i-maxlen:i+1]      #注意ans与maxlen的更新顺序不能颠倒
                maxlen += 1
        return ans
        # write code here
###################################################                                                                   |
#给定字符串A以及它的长度n，请返回最长回文子串的长度                                                                     |
#sol1:可类比上题（最长公共子串）idea，关键是需要注意更新maxlen的条件                                                    |sol2:中心扩散
#从头到尾扫描字符串，每次更新maxlen和ans后，第一个长度更长(maxlen+1 or maxlen+2)的回文子串只会以当前字符之后的字符为结尾 |
class Solution:                                                                                                       | class Solution: 
    def getLongestPalindrome(self, A, n):                                                                             |     def getLongestPalindrome(self, A, n):
        # write code here                                                                                             |         maxlen = 0
        max_len = 0                                                                                                   |         for i in range(n):
        for i in range(n):                                                                                            |             maxlen = max(temp,spread(A,i, i, n),spread(A,i, i+1, n)) #注意奇长度回文串与偶长度回文串的参数略有不同
            oddNum = A[i-max_len-1:i+1]                                                                               |         return 
            evenNum = A[i-max_len:i+1]                                                                                | def spread(A,left,right,n):        #以A[left]和A[right]为中心向左右两边扩散,返回扩散的最大长度
            if i-max_len-1>=0 and oddNum == oddNum[::-1]:                                                             |     while left >= 0 and right <= n-1 and A[left] == A[right]:  #利用回文子串正反一样的特点进行扩散，注意left和right的边界条件
                max_len+=2                                                                                            |         left -= 1
            elif i-max_len>=0 and evenNum == evenNum[::-1]:                                                           |         right += 1
                max_len+=1                                                                                            |     return right - left - 1
        return max_len
###################################################    
#根据快速排序的思路，找出数组中第K大的数
class Solution:
    def findKth(self, a, n, K):
        l = 0
        r = n - 1
        while l <= r:
            p = partion(a,l,r)
            if p+1 == K:
                return a[p]
            elif p+1 > K:
                r = p-1
            else:
                l = p+1
def partion(alist,l,r):
    num = alist[l]
    left = l + 1
    right = r
    while left <= right:
        while left <= right and alist[left] >= num:
            left += 1
        while right >= left and alist[right] <= num:
            right -= 1
        if left <= right:
            alist[left], alist[right] = alist[right], alist[left]
    alist[right],alist[l] = num,alist[right]
    return right
        # write code here
###################################################
#请写一个整数计算器，支持加减乘除运算和括号
# @param s string字符串 待计算的表达式
# @return int整型
class Solution:
    def solve(self , s ): #此题亦可采用先转为后缀表达式再求值
        partStack = []    #keyidea:用列表保存各部分的值,整个表达式的值即为各部分的和 (ex:原表达式形如(A*(B-C))*D,则列表保存(A*(B-C))的值和D的值)
        sign = '+'     #使用sign记录运算符,初始化为 '+'
        number = 0
        i = 0
        while i < len(s):
            if s[i] in '0123456789':
                number = number * 10 + ord(s[i]) - ord('0')  #number记录字符串中的数字部分
            if s[i] == '(':                               #遇到左括号时递归求这个括号里面的表达式的值
                counterPartition = 1
                j = i
                while counterPartition > 0:
                    j += 1
                    if s[j] == '(':
                        counterPartition += 1
                    if s[j] == ')':
                        counterPartition -= 1 #先遍历找到对应的右括号,counterPartition统计括号对数直到变量为0
                S = Solution()
                number = S.solve(s[i+1:j])    #利用递归来达到去括号的目的,去括号后的表达式具有A+B*C-D的形式
                i = j    #更新当前扫描到字符的索引
            if (s[i] in '+-*') or (i == len(s) - 1):  #遇到运算符时或者到表达式末尾时,说明该操作符之前的子表达式已计算完成，可根据sign的值作相应的运算
                if sign == '+':   #如果是当前sign为+,直接将当前number push 进去
                    partStack.append(number)
                if sign == '-':     #如果是-，push 进去-number
                    partStack.append(-number)
                if sign == '*':      #如果是 ×、÷ ，pop 出一个运算数和当前数作计算
                    partStack.append(partStack.pop() * number)
                if sign == '/':
                    partStack.append(partStack.pop() / number)
                sign = s[i]      #然后保存新的运算符到sign
                number = 0       #并将number置零,即完成一个部分的计算
            i += 1
        return sum(partStack)
        # write code here
###################################################
#给定一棵二叉树以及这棵树上的两个节点 o1 和 o2，请找到 o1 和 o2 的最近公共祖先节点。
#从根节点往下递归：
#1. 若该节点是第一个值为o1或o2的节点，则该节点是最近公共祖先；
#2. 否则，看左子树是否包含o1或o2：
    #2.1 若左子树包含o1或o2，则看右子树有没有：
        #a. 若右子树没有，则公共祖先在左子树
        #b. 若右子树有，则o1和o2肯定是左右子树一边一个，则公共祖先是根节点；
    #2.2 若左子树不包含o1和o2，则公共祖先在右子树或者该根子树不包含o1和o2。（两种情况都取决于右子树）

# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
 
# @param root TreeNode类 
# @param o1 int整型 
# @param o2 int整型 
# @return int整型
class Solution:
    def lowestCommonAncestor(self , root , o1 , o2 ):
        return self.dfs(root,o1,o2).val 
    # 该子树是否包含o1或o2
    def dfs(self,root,o1,o2):
        if root is None: return  None    
        if root.val == o1 or root.val == o2: return root    
        left = self.dfs(root.left,o1,o2)
        # 左子树没有，则在右子树
                # 若右子树没有,则右子树返回 None
        if left == None: return self.dfs(root.right,o1,o2)
        # 左子树有，则看右子树有没有
        right = self.dfs(root.right,o1,o2)
        if right == None: return left
        # 左子树右子树都有,则最近的祖先节点是root
        return root
        # write code here
################################################### 
#给定数组arr，输出arr的最长递增子序列。（如果有多个答案，请输出其中字典序最小的）
# retrun the longest increasing subsequence
# @param arr int整型一维数组 the array
# @return int整型一维数组
#
class Solution:
    def LIS(self , arr ):
        monoList = [arr[0]]      #长度为 i+1 的子序列的最后一位的最小值(不是解,只是长度关联),单调递增
        maxlen = [1]       #maxlen[i]记录以arr[i]结尾的最长递增子列长度
        for i in arr[1:]:
            if i >= monoList[-1]:
                monoList.append(i)
                maxlen.append(len(monoList))
            else:
                pos = findPosition(monoList, i)   #二分查找monoList中第一个 >= arr[i]的元素位置(必然存在)
                monoList[pos] = i              #找到位置后替换掉，而非插入
                maxlen.append(pos+1)             #monoList序列的总长不变，但是为了复用原序列一些 < arr[i]的结果，把arr[i]替换到合适的位置
        length = len(monoList)
        ans = [0]*length
        for j in range(-1,-len(arr)-1,-1): #倒着遍历arr,找到满足长度的maxlen就记录，然后更新。（即同样值的maxlen，选尽量靠右边的）
            if length > 0:
                if maxlen[j] == length:
                    ans[length-1] = arr[j]
                    length -= 1
            else:break
        return ans
def findPosition(li, t):     
    if li[-1] < t: return None
    l = 0
    r = len(li) - 1
    while l < r:
        mid = (l + r) >> 1
        if li[mid] >= t:
            r = mid
        else:
            l = mid + 1
    return r
        # write code here
###################################################
#给定一个二叉树和一个值sum，请找出所有的根节点到叶子节点的节点值之和等于sum的路径
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# @param root TreeNode类 
# @param sum int整型 
# @return int整型二维数组
#
class Solution:
    def pathSum(self , root , target ):
        if not root: return []
        paths = preOrder(root, [], [])
        return [p for p in paths if sum(p) == target]
def preOrder(root, path, paths):
    path.append(root.val)         #将当前节点值加入path
    if root.left:
        preOrder(root.left, path, paths)      #递归左子树
    if root.right:
        preOrder(root.right, path, paths)     #递归右子树
    if not root.left and not root.right:    #当递归到没有子树的时候就说明遍历完一条路径path
        paths.append(path.copy())
    path.pop()      #回溯很重要，从path1转到path2时须先移除path1最后一个元素，才能继续添加path2的节点
    return paths
        # write code here
###################################################
#给出n对括号，请编写一个函数来生成所有的由n对括号组成的合法组合
#关键：当前位置左括号不少于右括号
#构造图，其中节点：目前位置左括号和右括号数（x,y）(x>=y)  边：从（x,y)到（x+1,y）和（x，y+1），x==y时，没有（x,y+1）这条边
#解是从(0,0)出发到(n,n)的全部路径(类比上题)
# @param n int整型 
# @return string字符串一维数组
#
class Solution:
    def generateParenthesis(self , n ):
        return dfs(n, 0, 0, '', [])
def dfs(n, x, y, s, ans):
    if x == n and y == n: ans.append(s)
    if x < n: dfs(n, x+1, y, s+'(', ans)
    if x > y: dfs(n, x, y+1, s+')', ans)
    return ans
        # write code here
###################################################
#现在有一个只包含数字的字符串，将该字符串转化成IP地址的形式，返回所有可能的情况
# @param s string字符串 
# @return string字符串一维数组
#关于ip: 1.包含四段字符串，每一段≤”255”，≥0且至多包含3个字符；2.如果一个字符包括2个或3个字符，它的第一个字符不能为0，不能为010或者03等
class Solution:
    def restoreIpAddresses(self , s ):
        return dfs(s, [], [])
def dfs(s, tmp, ans):
    if not s and len(tmp) == 4: ans.append('.'.join(tmp))
    for i in range(1, len(s)+1):      #i为每一段的长度，可以为1，2，3
        if i > 1 and s[0] == '0': break   #判断是否合法
        if int(s[0:i]) > 255: break
        tmp.append(s[0:i])    #tmp列表保存每一步分割后的字符串
        dfs(s[i:], tmp, ans)  #递归
        tmp.pop()      #回溯
    return ans
        # write code here
###################################################
#在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。
#输入一个数组,求出这个数组中的逆序对的总数P
# -*- coding:utf-8 -*-
class Solution:
    def InversePairs(self, data):
        return inversePairsHelp(data, 0, len(data)-1)
def inversePairsHelp(data, l, r):    #借助归并排序
    if l == r: return 0
    mid = l + ((r - l) >> 1)
    left = inversePairsHelp(data, l, mid)
    right = inversePairsHelp(data, mid + 1, r)
    ans = left + right         #整个数组的逆序对 = 左子数组逆序对+右子数组逆序对 + 左右子数组共同组成逆序对  (递归)
    tmp = data[l:r+1]          #当左右数组有序时可在线性时间复杂度内得到第三项,考虑到归并排序也有递归的思想,因此可在一个递归下同时做这两件事情:排序和计算逆序对(O(nlogn))
    i, j, k = mid - l, r - l, r
    while i >= 0 and j >= mid+1-l:     #每次让data[l:r+1]中的最大数归位
        if tmp[i] > tmp[j]:
            ans += j - mid + l
            data[k] = tmp[i]
            i -= 1
        else:
            data[k] = tmp[j]
            j -= 1
        k -= 1
    if i >= 0:                      #让剩下的数归位
        data[l:k+1] = tmp[0:i+1]
    if j >= mid+1-l: 
        data[l:k+1] = tmp[mid+1-l:j+1]
    return ans
        # write code here
###################################################    
#有一个源源不断的吐出整数的数据流，假设你有足够的空间来保存吐出的数。请设计一个名叫MedianHolder的结构，MedianHolder可以随时取得之前吐出所有数的中位数。
#[要求]
#1. 如果MedianHolder已经保存了吐出的N个数，那么将一个新数加入到MedianHolder的过程，其时间复杂度是O(logN)。
#2. 取得已经吐出的N个数整体的中位数的过程，时间复杂度为O(1)
#keypoint:中位数即某个能将所有数根据大小关系划分为左右两半的数，只要能够不断的维护好这两部分的数，中位数就能在O(1)的时间内求得
# @param operations int整型二维数组 ops
# @return double浮点型一维数组
#
import heapq
class Solution:
    def __init__(self):
        self.maxHeap, self.minHeap = [], []    #maxHeap存放小于等于中位数, minHeap存放大于中位数
    def flowmedian(self , operations ):
        ans = []
        for op in operations:
            if op[0] == 1:
                self.addNum(op[1])
            if op[0] == 2:
                ans.append(self.findMedia())
        return ans
    def addNum(self, num):    #trick:让数据在两个堆中流动一遍
        heapq.heappush(self.maxHeap, -num)
        heapq.heappush(self.minHeap, -heapq.heappop(self.maxHeap))
        if len(self.maxHeap) < len(self.minHeap):       # 平衡左右两个堆的大小，总是保证len(maxHeap) >= len(minHeap)且至多多1
            heapq.heappush(self.maxHeap, -heapq.heappop(self.minHeap))
    def findMedia(self):
        if len(self.maxHeap) == 0: return -1
        if len(self.maxHeap) == len(self.minHeap): return (self.minHeap[0] - self.maxHeap[0]) / 2   #偶数个数据
        else: return -self.maxHeap[0]    #奇数个数据
        # write code here
###################################################    
#一个整型数组里除了两个数字只出现一次，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字 
# @param array int整型一维数组 
# @return int整型一维数组
#由于存在a != b, tmp == a ^ b且tmp不等于0, 依据tmp二进制表示中右起第一位不为1的位置将数组划分为两部分
#则a, b必分属不同部分，再次对两部分分别异或即得a, b
class Solution:
    def FindNumsAppearOnce(self , array ):
        ans = [0] * 2
        tmp = 0
        i = 1
        for num in array: tmp = tmp ^ num    # ^异或: a^0 = a, a^a = 0
        while tmp & i == 0:                  # &按位与
            i <<= 1 
        for num in array:
            if num & i == 0: ans[0] ^= num
            else: ans[1] ^= num
        if ans[0] > ans[1]: ans.reverse()
        return ans
        # write code here
###################################################
#把只包含质因子2、3和5的数称作丑数（Ugly Number）,习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数
class Solution:
    def GetUglyNumber_Solution(self, index):
        if not index: return 0
        uglys = [1]
        p2, p3, p5 = 0, 0, 0
        for i in range(1, index): #前面生成的每一个丑数都应该乘以2，3，5以产生新的丑数
            uglys.append(min(uglys[p2] * 2, uglys[p3] * 3 ,uglys[p5] * 5))#但是我们不必比较前面的每一个丑数, 例如假设前面的某个丑数a 因为乘以2 得到了当前的丑数b，下一次再生成下一个丑数时，就不用再考察a以及a前面的丑数*2的情况
            if uglys[i] % 2 == 0: p2 += 1
            if uglys[i] % 3 == 0: p3 += 1
            if uglys[i] % 5 == 0: p5 += 1
        return uglys[-1]
        # write code here
###################################################
#一个重复字符串是由两个相同的字符串首尾拼接而成，例如abcabc便是长度为6的一个重复字符串，而abcba则不存在重复字符串。
#给定一个字符串，请编写一个函数，返回其最长的重复字符子串。若不存在任何重复字符子串，则返回0。
# @param a string字符串 待计算字符串
# @return int整型
#
class Solution:
    def solve(self , a ):
        n = len(a)
        if n < 2:
            return 0
        count = 0
        for window in range(n // 2, -1, -1):
            for j in range(n - window):
                if a[j] == a[j+window]:
                    count += 1
                else:
                    count = 0
                if count == window:
                    return 2 * window
        return 0
        # write code here
###################################################
#在一个二维数组array中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序
#请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param target int整型 
# @param array int整型二维数组 
# @return bool布尔型
#
class Solution:
    def Find(self , target: int, array: List[List[int]]) -> bool:
        end_row, start_col = len(array) - 1, 0
        return FindHelp(target, array, end_row, start_col)
def FindHelp(target, array, end_row, start_col):
    #IMPORTANT: 基于array的特点,从左下角(记为ld)开始查找
    #当target=ld,直接返回True; 当target>ld,只须在ld所在列右方查找; 当target<ld只需在ld所在行上方查找,递归
    if start_col >= len(array[0]) or end_row < 0: return False    #超出范围还未找到,返回False
    if target == array[end_row][start_col]: return True
    if target > array[end_row][start_col]: 
        return FindHelp(target, array, end_row, start_col + 1)
    if target < array[end_row][start_col]:
        return FindHelp(target, array, end_row - 1, start_col)
        # write code here
###################################################
#给定两个递增数组arr1和arr2，已知两个数组的长度都为N，求两个数组中所有数的上中位数(time:O(logN),space:O(1))
#note:上中位数：假设递增序列长度为n，为第n/2个数
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# find median in two sorted array
# @param arr1 int整型一维数组 the array1
# @param arr2 int整型一维数组 the array2
# @return int整型
#
class Solution:
    def findMedianinTwoSortedAray(self , arr1: List[int], arr2: List[int]) -> int:
        return aux(arr1, arr2, 0, len(arr1)-1, 0, len(arr2)-1)
def aux(arr1, arr2, l1, r1, l2, r2):
    if l1 == r1: return min(arr1[l1], arr2[l2])     #终止条件1
    mid1 = (l1 + r1) // 2
    mid2 = (l2 + r2) // 2
    lenth = r1 - l1 + 1
    if arr1[mid1] == arr2[mid2]:       #终止条件2
        return arr1[mid1]
    if arr1[mid1] > arr2[mid2]:
        r1 = mid1
        l2 = mid2 if lenth % 2 else mid2+1          #IMPORTANT:数组长度分奇偶性讨论, 确保两个数组删除的元素个数相同
    if arr1[mid1] < arr2[mid2]:
        r2 = mid2
        l1 = mid1 if lenth % 2 else mid1+1
    return aux(arr1, arr2, l1, r1, l2, r2)          #递归求解
        # write code here
###################################################
#现在有一个没有重复元素的整数集合S，求S的所有子集
#注意：给出的子集中的元素必须按升序排列; 给出的解集中不能出现重复的元素
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param S int整型一维数组 
# @return int整型二维数组
#
class Solution:
    def subsets(self , S: List[int]) -> List[List[int]]:
        if not S: return []
        S.sort()      #排序,满足子集内升序
        allsubs = []
        for lenth in range(len(S)+1):     #以子集长度分类添加
            dfs(S, [], allsubs, 0, lenth)
        return allsubs
def dfs(S, tmpSet, res, start, Len):     #更新start参数确保不会有重复集合
    if len(tmpSet) == Len: res.append(tmpSet.copy())      #满足长度条件，加入res
    else:
        for i in range(start, len(S)):
            tmpSet.append(S[i])
            dfs(S, tmpSet, res, i+1, Len)
            tmpSet.pop()
    return res
        # write code here
###################################################
#给定一个非负整数 n ，返回 n! 结果的末尾为 0 的数量
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# the number of 0
# @param n long长整型 the number
# @return long长整型
#
#由因数分解知识可将问题转为: n!能拆出多少对(2,5)
#IMPORTANT: 可以发现，有5因子的数比有2因子的数要少,所以只需求能拆出来多少个因子5即可(因为一定能有足够数量的因子2来匹配)
#又发现，5的倍数可以至少产生1个5，25的倍数可以产生至少2个5，125的倍数可以产生至少3个5...，因此答案即为 n/5+n/25+n/25...
class Solution:
    def thenumberof0(self , n: int) -> int:
        res = 0
        div = 5
        while n // div:
            res += n // div
            div *= 5
        return res
        # write code here
###################################################
#给定一个长度为n的数组nums，请你找到峰值并返回其索引。数组可能包含多个峰值，在这种情况下，返回任何一个所在位置即可。
#1.峰值元素是指其值严格大于左右相邻值的元素。严格大于即不能有等于
#2.假设 nums[-1] = nums[n] = −∞
#3.对于所有有效的 i 都有 nums[i] != nums[i + 1]
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
# @param nums int整型一维数组 
# @return int整型
#最优解法：二分
class Solution:
    def findPeakElement(self , nums: List[int]) -> int:
        l = 0
        r = len(nums) - 1     #[l,r]为一定存在峰值的区间
        while l < r:          #不断二分压缩区间,最后的单点集即为峰值索引
            mid = (l + r) >> 1
            if nums[mid] < nums[mid+1]:
                l = mid + 1     #IMPORTANT: l, r的更新必须保证[l,r]一定存在峰值
            else:
                r = mid
        return l
        # write code here
