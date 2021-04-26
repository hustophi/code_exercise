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
###################################################
