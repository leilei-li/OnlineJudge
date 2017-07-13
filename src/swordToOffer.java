/**
 * Created by lileilei on 2017/7/10.
 */

import java.util.*;

public class swordToOffer {

    private class ListNode {
        int val;
        ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }
    }

    private class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    public boolean Find(int target, int[][] array) {
        int row = array.length;
        int col = array[0].length;
        int targetRow = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (array[i][j] == target) return true;
            }
        }
        return false;
    }

    public String replaceSpace(StringBuffer str) {
        StringBuffer result = new StringBuffer();
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            if (c != ' ') result.append(c);
            else result.append("%20");
        }
        return result.toString();
    }

    public int Fibonacci(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1;
        else return Fibonacci(n - 1) + Fibonacci(n - 2);
    }

    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> result = new ArrayList<>();
        if (listNode == null) return result;
        Stack<Integer> stack = new Stack<>();
        while (listNode != null) {
            stack.push(listNode.val);
            listNode = listNode.next;
        }
        while (stack.isEmpty() == false) {
            result.add(stack.pop());
        }
        return result;
    }

    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        TreeNode root = reConstructBinaryTree2(pre, 0, pre.length - 1,
                in, 0, in.length - 1);
        return root;
    }

    private TreeNode reConstructBinaryTree2(int[] pre, int startPre, int endPre, int[] in,
                                            int startIn, int endIn) {
        if (startPre > endPre || startIn > endIn) return null;
        TreeNode root = new TreeNode(pre[startPre]);
        for (int i = startIn; i <= endIn; i++) {
            if (in[i] == pre[startPre]) {
                root.left = reConstructBinaryTree2(pre, startPre + 1,
                        startPre + i - startIn, in, startIn, i - 1);
                root.right = reConstructBinaryTree2(pre, i - startIn + startPre + 1,
                        endPre, in, i + 1, endIn);
            }
        }
        return root;
    }

    public int minNumberInRotateArray(int[] array) {
        int n = array.length;
        if (n == 0) return 0;
        if (n == 1) return array[0];
        for (int i = 1; i < n; i++) {
            if (array[i - 1] > array[i]) return array[i];
        }
        return 1;
    }

    public int RectCover(int target) {
        if (target == 0) return 0;
        if (target == 1) return 1;
        if (target == 2) return 2;
        return RectCover(target - 1) + RectCover(target - 2);
    }

    public int NumberOf1(int n) {
        int result = Integer.toBinaryString(n).replace("0", "").length();
        return result;
    }

    public double Power(double base, int exponent) {
        if (exponent == 0) return 1.0;
        double result = 1.0;
        for (int i = 0; i < Math.abs(exponent); i++) {
            result = result * base;
        }
        if (exponent > 0) return result;
        else return (double) 1.0 / result;
    }

    public void reOrderArray(int[] array) {
        int[] result = new int[array.length];
        int position = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] % 2 == 1) {
                result[position] = array[i];
                position++;
            }
        }
        for (int i = 0; i < array.length; i++) {
            if (array[i] % 2 == 0) {
                result[position] = array[i];
                position++;
            }
        }
        System.arraycopy(result, 0, array, 0, array.length);
    }

    public ListNode FindKthToTail(ListNode head, int k) {
        if (head == null || k <= 0) return null;
        ListNode fast, slow;
        fast = slow = head;
        for (int i = 1; i < k; i++) {
            if (fast.next != null) fast = fast.next;
            else return null;
        }
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }

    public ListNode ReverseList(ListNode head) {
        if (head == null || head.next == null) return head;
        Stack<ListNode> stack = new Stack<>();
        ListNode cur = head;
        while (cur.next != null) {
            stack.push(cur);
            cur = cur.next;
        }
        ListNode newHead = cur;
        while (!stack.isEmpty()) {
            cur.next = stack.pop();
            cur = cur.next;
        }
        cur.next = null;
        return newHead;
    }

    public ListNode Merge(ListNode list1, ListNode list2) {
        ListNode head = new ListNode(0);
        ListNode cur = head;
        while (list1 != null && list2 != null) {
            if (list1.val <= list2.val) {
                cur.next = list1;
                list1 = list1.next;
            } else {
                cur.next = list2;
                list2 = list2.next;
            }
            cur = cur.next;
        }
        if (list1 == null) {
            cur.next = list2;
        }
        if (list2 == null) {
            cur.next = list1;
        }
        return head.next;
    }

    public boolean HasSubtree(TreeNode root1, TreeNode root2) {
        boolean result = false;
        if (root1 != null && root2 != null) {
            if (root1.val == root2.val) result = isSubTree(root1, root2);
            if (result == false) result = isSubTree(root1.left, root2);
            if (result == false) result = isSubTree(root1.right, root2);
        }
        return result;
    }

    private boolean isSubTree(TreeNode node1, TreeNode node2) {
        if (node2 == null) return true;
        if (node1 == null) return false;
        if (node1.val != node2.val) return false;
        return isSubTree(node1.left, node2.left) && isSubTree(node1.right, node2.right);
    }

    public void Mirror(TreeNode root) {
        if (root == null) return;
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        Mirror(root.left);
        Mirror(root.right);
    }

    public ArrayList<Integer> printMatrix(int[][] matrix) {
        ArrayList<Integer> result = new ArrayList<>();
        int row = matrix.length;
        int col = matrix[0].length;
        int circle = ((row < col ? row : col) - 1) / 2 + 1;
        for (int i = 0; i < circle; i++) {
            for (int j = i; j < col - i; j++) {
                result.add(matrix[i][j]);
            }
            for (int k = i + 1; k < row - i; k++) {
                result.add(matrix[k][col - 1 - i]);
            }
            for (int m = col - i - 2; (m >= i) && (row - i - 1 != i); m--) {
                result.add(matrix[row - i - 1][m]);
            }
            for (int n = row - i - 2; (n > i) && (col - i - 1 != i); n--) {
                result.add(matrix[n][i]);
            }
        }
        return result;
    }

    public boolean IsPopOrder(int[] pushA, int[] popA) {
        if (pushA.length == 0) return false;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0, j = 0; i < pushA.length; i++) {
            stack.push(pushA[i]);
            while (j < popA.length && stack.peek() == popA[j]) {
                stack.pop();
                j++;
            }
        }
        return stack.isEmpty();
    }

    public int JumpFloor(int target) {
        return step(target);
    }

    private int step(int n) {
        if (n == 1) {
            return 1;
        }
        if (n == 2) {
            return 2;
        } else return step(n - 1) + step(n - 2);
    }

    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        if (root == null) return result;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node.left != null) queue.offer(node.left);
            if (node.right != null) queue.offer(node.right);
            result.add(node.val);
        }
        return result;
    }

    public boolean VerifySquenceOfBST(int[] sequence) {
        if (sequence.length == 0) return false;
        if (sequence.length == 1) return true;
        return isBST(sequence, 0, sequence.length - 1);
    }

    private boolean isBST(int[] a, int start, int root) {
        if (start >= root) return true;
        int i = root;
        while (i > start && a[i - 1] > a[root]) i--;
        for (int j = start; j < i - 1; j++) {
            if (a[j] > a[root]) return false;
        }
        return isBST(a, start, i - 1) && isBST(a, i, root - 1);
    }


}