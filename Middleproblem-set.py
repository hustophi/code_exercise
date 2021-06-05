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
    for i in range(1, len(s)+1):
        if i > 1 and s[0] == '0': break   #判断是否合法
        if int(s[0:i]) > 255: break
        tmp.append(s[0:i])    #tmp列表保存每一步分割后的字符串
        dfs(s[i:], tmp, ans)
        tmp.pop()      #回溯
    return ans
        # write code here
