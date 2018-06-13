import java.util.*;
import java.math.*;

public class LeetCode3 {
    private class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    public int run(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return 1;
        }
        if (root.left == null) {
            return run(root.right) + 1;
        }
        if (root.right == null) {
            return run(root.left) + 1;
        }
        return Math.min(run(root.left), run(root.right)) + 1;
    }

    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> hashMap = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            hashMap.put(target - nums[i], i);
        }
        int[] result = new int[2];
        for (int i = 0; i < nums.length; i++) {
            int result1 = nums[i];
            if (hashMap.containsKey(target - result1)) {
                result[0] = i + 1;
                result[1] = hashMap.get(target - result1) + 1;
                break;
            }
        }
        return result;
    }
}
