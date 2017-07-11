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

    


}