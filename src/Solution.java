import java.util.Stack;

/**
 * Created by lileilei on 2017/3/31.
 */
public class Solution {
    public class TreeNode {
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

}
