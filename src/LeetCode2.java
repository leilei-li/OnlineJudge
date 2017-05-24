/**
 * Created by lileilei on 2017/5/11.
 */

import java.util.*;

public class LeetCode2 {

    private class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    private class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
            next = null;
        }
    }

    public void setZeroes(int[][] matrix) {
        boolean hasZeroInRow = false;
        boolean hasZeroInCol = false;
        int row = matrix.length;
        int col = matrix[0].length;
        for (int i = 0; i < row; i++) {
            if (matrix[i][0] == 0) {
                hasZeroInCol = true;
                break;
            }
        }
        for (int i = 0; i < col; i++) {
            if (matrix[0][i] == 0) {
                hasZeroInRow = true;
                break;
            }
        }
        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                if (matrix[i][j] == 0) {
                    matrix[0][j] = 0;
                    matrix[i][0] = 0;
                }
            }
        }
        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        if (hasZeroInCol) {
            for (int i = 0; i < row; i++) {
                matrix[i][0] = 0;
            }
        }
        if (hasZeroInRow) {
            for (int i = 0; i < col; i++) {
                matrix[0][i] = 0;
            }
        }
    }

    public int minDistance(String word1, String word2) {
        if (word1.length() == 0 && word2.length() == 0) return 0;
        if (word1.length() == 0) return word2.length();
        if (word2.length() == 0) return word1.length();
        int[][] dp = new int[word1.length() + 1][word2.length() + 1];
        dp[0][0] = 0;
        for (int i = 1; i <= word1.length(); i++) {
            dp[i][0] = i;
        }
        for (int i = 1; i <= word2.length(); i++) {
            dp[0][i] = i;
        }
        for (int i = 1; i <= word1.length(); i++) {
            for (int j = 1; j <= word2.length(); j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) dp[i][j] = dp[i - 1][j - 1];
                else dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
            }
        }
        return dp[word1.length()][word2.length()];
    }
}
