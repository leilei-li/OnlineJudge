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

}
