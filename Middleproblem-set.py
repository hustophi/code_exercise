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
