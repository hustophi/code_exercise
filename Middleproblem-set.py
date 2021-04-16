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
