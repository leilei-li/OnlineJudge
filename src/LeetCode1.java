import java.util.*;

/**
 * Created by lileilei on 2017/4/28.
 */
public class LeetCode1 {

    private class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    public int run(TreeNode root) {
        if (root == null)
            return 0;
        if (root.left == null && root.right == null)
            return 1;
        if (root.left == null)
            return run(root.right) + 1;
        if (root.right == null)
            return run(root.left) + 1;
        return Math.min(run(root.left), run(root.right)) + 1;
    }

    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<Integer>();
        int a, b;
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i].equals("+")) {
                b = stack.pop();//注意栈是先进后出
                a = stack.pop();
                stack.push(a + b);
            } else if (tokens[i].equals("-")) {
                b = stack.pop();//注意栈是先进后出
                a = stack.pop();
                stack.push(a - b);
            } else if (tokens[i].equals("*")) {
                b = stack.pop();//注意栈是先进后出
                a = stack.pop();
                stack.push(a * b);
            } else if (tokens[i].equals("/")) {
                b = stack.pop();//注意栈是先进后出
                a = stack.pop();
                stack.push(a / b);
            } else {
                int num = Integer.parseInt(tokens[i]);
                stack.push(num);
            }
        }
        return stack.pop();
    }

    private class Point {
        int x;
        int y;

        Point() {
            x = 0;
            y = 0;
        }

        Point(int a, int b) {
            x = a;
            y = b;
        }
    }

    public int maxPoints(Point[] points) {
        int result = 0;
        int n = points.length;
        if (n < 2) return n;
        for (int i = 0; i < n; i++) {
            int dup = 1;//与点a重合的点，即x，y均相等的点,起始值为1
            int vtl = 0;//与点a在同一x上的点，即x相同而y不同的点
            HashMap<Float, Integer> map = new HashMap<Float, Integer>();
            //用来保存斜率相同的点有多少个，每个循环建立一个
            Point a = points[i];
            for (int j = 0; j < n; j++) {//穷举所有的点
                if (i == j) continue;//本身跳过不计
                Point b = points[j];
                if (a.x == b.x) {
                    if (a.y == b.y) dup++;
                    else vtl++;
                } else {
                    float k = (float) (a.y - b.y) / (a.x - b.x);//计算斜率，记得要格式转化成float
                    if (map.containsKey(k)) {
                        map.put(k, map.get(k) + 1);
                    } else {
                        map.put(k, 1);
                    }//保存斜率相同的点
                }
            }
            int max = vtl;//斜率相同但是若y不同会是两条直线
            for (float k : map.keySet()) {
                max = Math.max(max, map.get(k));//找寻斜率相同的最多点
            }
            result = Math.max(result, max + dup);//斜率相同的点+重合最多的点
        }
        return result;
    }

    private class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
            next = null;
        }
    }

    public ListNode sortList(ListNode head) {
        listQuickSort(head, null);
        return head;
    }

    private void listQuickSort(ListNode head, ListNode end) {
        if (head != end) {
            ListNode quickSort = quickSort(head);
            listQuickSort(head, quickSort);
            listQuickSort(quickSort.next, end);
        }
    }

    private ListNode quickSort(ListNode head) {
        ListNode slow = head;
        ListNode fast = head.next;
        while (fast != null) {
            if (fast.val < head.val) {
                slow = slow.next;
                int temp = slow.val;
                slow.val = fast.val;
                fast.val = temp;
            }
            fast = fast.next;
        }
        int temp = slow.val;
        slow.val = head.val;
        head.val = temp;
        return slow;

    }

    public ListNode insertionSortList(ListNode head) {
        ListNode listNode = new ListNode(Integer.MIN_VALUE);
        ListNode cur = head;
        ListNode pre = listNode;
        //用最小值做表头，新建一个链表
        while (cur != null) {
            ListNode next = cur.next;
            pre = listNode;
            //重置插入点到链表头
            while (pre.next != null && pre.next.val < cur.val) {
                pre = pre.next;//要插入的值cur要移到左边比他小而右边比他大的位置
            }
            //到达指定位置,链表插入相关值,就是普通链表插入
            cur.next = pre.next;
            pre.next = cur;
            cur = next;//处理下一个要插入的值
        }
        return listNode.next;
    }

    public ArrayList<Integer> postorderTraversal(TreeNode root) {
        TreeNode cur = root;
        TreeNode pre = null;
        Stack<TreeNode> stack = new Stack<TreeNode>();
        ArrayList<Integer> list = new ArrayList<Integer>();
        while (cur != null || stack.isEmpty() == false) {
            if (cur != null) {
                stack.push(cur);
                cur = cur.left;//左孩子入栈
            } else {
                cur = stack.peek();
                cur = cur.right;//左孩子空了后，从栈顶元素右孩子开始重复左孩子入栈过程
                if (cur != null && cur != pre) {
                    stack.push(cur);
                    cur = cur.left;
                } else {
                    cur = stack.pop();
                    list.add(cur.val);
                    pre = cur;
                    cur = null;
                }
            }
        }
        return list;
    }

    public ArrayList<Integer> preorderTraversal(TreeNode root) {
        ArrayList<Integer> list = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = null;
        if (root == null) return list;
        stack.push(root);
        while (stack.isEmpty() == false) {//使用栈实现前序遍历
            cur = stack.pop();
            list.add(cur.val);
            if (cur.right != null) stack.push(cur.right);//栈是先进后出，所以先右后左
            if (cur.left != null) stack.push(cur.left);
        }
        return list;
    }

    public void reorderList(ListNode head) {
        if (head == null || head.next == null) return;
        ListNode fast, slow, mid;//快慢指针，找中点mid
        fast = slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }//fast到尾部，slow到中点
        mid = slow;
        ListNode preCur = slow.next;
        while (preCur.next != null) {
            ListNode cur = preCur.next;
            preCur.next = cur.next;
            cur.next = mid.next;
            mid.next = cur;
        }//逆转后半段链表
        ListNode a = head;
        ListNode b = mid.next;
        while (a != mid) {
            mid.next = b.next;
            b.next = a.next;
            a.next = b;
            a = b.next;
            b = mid.next;
        }//后半段插入前半段
    }

    public boolean hasCycle(ListNode head) {
        ListNode slow, fast;
        slow = fast = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) return true;
        }
        return false;
    }

    public ListNode detectCycle(ListNode head) {
        ListNode slow, fast;
        slow = fast = head;
        if (head == null || head.next == null) return null;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                slow = head;
                while (slow != fast) {
                    slow = slow.next;
                    fast = fast.next;
                }
                return slow;
            }
        }
        return null;
    }

    public boolean wordBreak(String s, Set<String> dict) {
        int len = s.length();
        boolean[] dp = new boolean[len + 1];//dp[i]表示s中0到i可分
        dp[0] = true;
        for (int i = 1; i <= len; i++)
            for (int j = 0; j < i; j++) {
                if (dp[j] && dict.contains(s.substring(j, i))) {
                    dp[i] = true;
                }
            }
        return dp[len];
    }

    public ArrayList<String> wordBreak2(String s, Set<String> dict) {
        ArrayList<String>[] dp = new ArrayList[s.length() + 1];
        dp[0] = new ArrayList<String>();
        for (int i = 0; i < s.length(); i++) {
            if (dp[i] == null) continue;//必须保证前面已经匹配过了
            for (String word : dict) {
                int len = word.length();
                int end = i + len;
                if (end > s.length()) continue;
                if (s.substring(i, end).equals(word)) {
                    if (dp[end] == null) {
                        dp[end] = new ArrayList<String>();
                    }
                    dp[end].add(word);
                }
            }
        }
        ArrayList<String> ans = new ArrayList<String>();
        if (dp[s.length()] == null) return ans;
        ArrayList<String> tmp = new ArrayList<String>();
        dfsSearch(dp, s.length(), ans, tmp);
        return ans;
    }

    private void dfsSearch(ArrayList<String>[] dp, int end, ArrayList<String> result, ArrayList<String> tmp) {
        if (end<=0){
            String ans=tmp.get(tmp.size()-1);
            for (int i = tmp.size()-2; i >=0 ; i--) {
                ans=ans+(" "+tmp.get(i));
            }
            result.add(ans);
            return;
        }
        for (String str:dp[end]){
            tmp.add(str);
            dfsSearch(dp,end-str.length(),result,tmp);
            tmp.remove(tmp.size()-1);
        }
    }
}
