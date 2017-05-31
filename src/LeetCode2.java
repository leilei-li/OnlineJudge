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

    public String simplifyPath(String path) {
        LinkedList<String> queue = new LinkedList<>();
        StringBuilder result = new StringBuilder();
        String[] str = path.split("/");
        for (int i = 0; i < str.length; i++) {
            if (str[i].equals("") || str[i].equals(".")) continue;
            if (str[i].equals("..")) {
                if (queue.isEmpty() == false) {
                    queue.pollLast();
                }
            } else queue.add(str[i]);
        }
        while (queue.isEmpty() == false) {
            result.append("/");
            result.append(queue.pollFirst());
        }
        if (result.length() == 0) return "/";
        return result.toString();
    }

    public int climbStairs(int n) {
        int[] dp = new int[n + 1];
        if (n == 1) return 1;
        if (n == 2) return 2;
        dp[0] = 0;
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 2] + dp[i - 1];
        }
        return dp[n];
    }

    public int sqrt(int x) {
        if (x == 0) return 0;
        double pre = 0;
        double cur = 1;
        while (Math.abs(cur - pre) > 0.001) {
            pre = cur;
            cur = pre / 2 + x / (2 * pre);
        }
        return (int) cur;
    }

    public ArrayList<String> fullJustify(String[] words, int L) {
        int i = 0;
        ArrayList<String> result = new ArrayList<>();
        while (i < words.length) {
            int j = i + 1;
            int len = words[i].length();
            while (j < words.length && len + 1 + words[j].length() <= L) {
                len = len + words[j].length() + 1;
                j++;
            }
            StringBuilder str = new StringBuilder();
            str.append(words[i]);
            if (j == words.length) {
                for (int k = i + 1; k < words.length; k++) {
                    str.append(" ");
                    str.append(words[k]);
                }
                while (str.length() < L) {
                    str.append(" ");
                }
            } else {
                int extraSpace = L - len;
                int spaceNum = j - i - 1;
                if (spaceNum == 0) {
                    while (str.length() < L) {
                        str.append(" ");
                    }
                } else {
                    for (int k = i + 1; k < j; k++) {
                        str.append(" ");
                        for (int l = 0; l < extraSpace / spaceNum; l++) {
                            str.append(" ");
                        }
                        if (k - i <= extraSpace % spaceNum) {
                            str.append(" ");
                        }
                        str.append(words[k]);
                    }
                }
            }
            result.add(str.toString());
            i = j;
        }
        return result;
    }

    public int[] plusOne(int[] digits) {
        int length = digits.length;
        digits[length - 1]++;
        for (int i = length - 1; i >= 1 && digits[i] >= 10; i--) {
            digits[i - 1]++;
            digits[i] = 0;
        }
        if (digits[0] < 10) return digits;
        int[] newDigits = new int[length + 1];
        newDigits[0] = 1;
        return newDigits;
    }

    public boolean isNumber(String s) {
        try {
            char c = s.charAt(s.length() - 1);
            if (c == 'f' || c == 'F' || c == 'd' || c == 'D') return false;
            Double.valueOf(s);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    public String addBinary(String a, String b) {
        char[] sum = new char[a.length() > b.length() ? a.length() : b.length()];
        int i = a.length() - 1;
        int j = b.length() - 1;
        int k = sum.length - 1;
        for (; i >= 0 && j >= 0; i--, j--, k--) {
            sum[k] = (char) ((a.charAt(i) - '0') + (b.charAt(j) - '0') + '0');
        }
        while (i >= 0) {
            sum[k--] = a.charAt(i--);
        }
        while (j >= 0) {
            sum[k--] = b.charAt(j--);
        }//进行补位
        for (int l = sum.length - 1; l > 0; l--) {
            int x = (sum[l] - '0') % 2;
            int carry = (sum[l] - '0') / 2;
            sum[l] = (char) ('0' + x);
            sum[l - 1] = (char) (sum[l - 1] + carry);
        }//处理进位
        if (sum[0] == '2' || sum[0] == '3') {
            char[] newSum = new char[sum.length + 1];
            System.arraycopy(sum, 1, newSum, 2, sum.length - 1);
            newSum[1] = (char) ((sum[0] - '0') % 2 + '0');
            newSum[0] = (char) ((sum[0] - '0') / 2 + '0');
            return new String(newSum);
        }
        return new String(sum);
    }
}
