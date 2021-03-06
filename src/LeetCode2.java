/**
 * Created by lileilei on 2017/5/11.
 */

import java.util.*;
import java.math.*;

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

    private class Interval {
        int start;
        int end;

        Interval() {
            start = 0;
            end = 0;
        }

        Interval(int s, int e) {
            start = s;
            end = e;
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

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode head = new ListNode(0);
        ListNode cur = head;
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                cur.next = l1;
                l1 = l1.next;
            } else {
                cur.next = l2;
                l2 = l2.next;
            }
            cur = cur.next;
        }
        if (l1 == null) {
            cur.next = l2;
        }
        if (l2 == null) {
            cur.next = l1;
        }
        return head.next;
    }

    public int minPathSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < m; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for (int i = 1; i < n; i++) {
            dp[0][i] = dp[0][i - 1] + grid[0][i];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[m - 1][n - 1];
    }

    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        dp[0][0] = 1;
        for (int i = 1; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i < n; i++) {
            dp[0][i] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            if (obstacleGrid[i][0] == 1) break;
            else dp[i][0] = 1;
        }
        for (int i = 0; i < n; i++) {
            if (obstacleGrid[0][i] == 1) break;
            else dp[0][i] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 1) dp[i][j] = 0;
                else dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    public ListNode rotateRight(ListNode head, int n) {
        if (head == null) return null;
        ListNode tmp = head;
        int len = 0;
        while (tmp != null) {
            len++;
            tmp = tmp.next;
        }
        n = n % len;
        if (n == 0) return head;
        ListNode cur, fast, slow;
        cur = fast = slow = head;
        for (int i = 0; i < n; i++) {
            if (fast != null) fast = fast.next;
            else return null;
        }
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        ListNode newHead = slow.next;
        slow.next = null;
        fast.next = cur;
        return newHead;
    }

    public String getPermutation(int n, int k) {
        k--;
        int[] num = new int[n];
        int cnt = 1;
        for (int i = 0; i < n; i++) {
            num[i] = i + 1;
            cnt = cnt * (i + 1);
        }
        char[] result = new char[n];
        for (int i = 0; i < n; i++) {
            cnt = cnt / (n - i);
            int p = k / cnt;
            result[i] = (char) ('0' + num[p]);
            for (int j = p; j < n - 1 - i; j++) {
                num[j] = num[j + 1];
            }
            k = k % cnt;
        }
        return new String(result);
    }

    public ArrayList<Integer> spiralOrder(int[][] matrix) {
        ArrayList<Integer> res = new ArrayList<>();
        if (matrix.length == 0) return res;
        int m = matrix.length, n = matrix[0].length;
        // 计算圈数
        int lvl = (Math.min(m, n) + 1) / 2;
        for (int i = 0; i < lvl; i++) {
            int lastRow = m - i - 1;
            int lastCol = n - i - 1;
            if (i == lastRow) {
                for (int j = i; j <= lastCol; j++) {
                    res.add(matrix[i][j]);
                }
            } else if (i == lastCol) {
                for (int j = i; j <= lastRow; j++) {
                    res.add(matrix[j][i]);
                }
            } else {
                for (int j = i; j < lastCol; j++) {
                    res.add(matrix[i][j]);
                }
                for (int j = i; j < lastRow; j++) {
                    res.add(matrix[j][lastCol]);
                }
                for (int j = lastCol; j > i; j--) {
                    res.add(matrix[lastRow][j]);
                }
                for (int j = lastRow; j > i; j--) {
                    res.add(matrix[j][i]);
                }
            }
        }
        return res;
    }

    public int[][] generateMatrix(int n) {
        int[][] res = new int[n][n];
        int left = 0, right = n - 1, bottom = n - 1, top = 0, num = 1;
        while (left < right && top < bottom) {
            for (int i = left; i < right; i++) {
                res[top][i] = num++;
            }
            for (int i = top; i < bottom; i++) {
                res[i][right] = num++;
            }
            for (int i = right; i > left; i--) {
                res[bottom][i] = num++;
            }
            for (int i = bottom; i > top; i--) {
                res[i][left] = num++;
            }
            top++;
            bottom--;
            left++;
            right--;
        }
        if (n % 2 == 1) {
            res[n / 2][n / 2] = num;
        }
        return res;
    }

    public int lengthOfLastWord(String s) {
        String[] str = s.split(" ");
        if (str.length == 0) return 0;
        return str[str.length - 1].length();
    }

    public ArrayList<Interval> insert(ArrayList<Interval> intervals, Interval newInterval) {
        if (intervals == null || newInterval == null) return intervals;
        if (intervals.size() == 0) {
            intervals.add(newInterval);
            return intervals;
        }
        ListIterator<Interval> iterator = intervals.listIterator();
        while (iterator.hasNext()) {
            Interval tmpInterval = iterator.next();
            if (newInterval.end < tmpInterval.start) {
                iterator.previous();
                iterator.add(newInterval);
                return intervals;
            } else {
                if (tmpInterval.end < newInterval.start) continue;
                else {
                    newInterval.start = Math.min(tmpInterval.start, newInterval.start);
                    newInterval.end = Math.max(tmpInterval.end, newInterval.end);
                    iterator.remove();
                }
            }
        }
        intervals.add(newInterval);
        return intervals;
    }

    private class MyComparator implements Comparator<Interval> {
        @Override
        public int compare(Interval a, Interval b) {
            return a.start - b.start;
        }
    }

    public ArrayList<Interval> merge(List<Interval> intervals) {
        ArrayList<Interval> ans = new ArrayList<Interval>();
        if (intervals.size() == 0) return ans;

        Collections.sort(intervals, new MyComparator());

        int start = intervals.get(0).start;
        int end = intervals.get(0).end;

        for (int i = 0; i < intervals.size(); i++) {
            Interval inter = intervals.get(i);
            if (inter.start > end) {
                ans.add(new Interval(start, end));
                start = inter.start;
                end = inter.end;
            } else {
                end = Math.max(end, inter.end);
            }
        }
        ans.add(new Interval(start, end));
        return ans;
    }

    public boolean canJump(int[] A) {
        if (A.length == 0) return false;
        boolean[] dp = new boolean[A.length];
        dp[0] = true;
        for (int i = 0; i < A.length; i++) {
            if (dp[i] == true) {
                for (int j = 1; j <= A[i]; j++) {
                    if (i + j < A.length) dp[i + j] = true;
                    else return true;
                }
            }
        }
        return dp[A.length - 1];
    }

    public int jump(int[] A) {
        int[] dp = new int[A.length];
        Arrays.fill(dp, 0);
        for (int i = 0; i < A.length; i++) {
            int maxStep = Math.min(i + A[i], A.length - 1);
            for (int j = i + 1; j <= maxStep; j++) {
                if (dp[j] == 0) dp[j] = dp[i] + 1;
            }
            if (dp[A.length - 1] != 0) break;
        }
        return dp[A.length - 1];
    }

    public int maxSubArray(int[] A) {
        int sum = 0;
        int max = A[0];
        for (int i = 0; i < A.length; i++) {
            sum = sum + A[i];
            max = Math.max(sum, max);
            if (sum < 0) sum = 0;
        }
        return max;
    }

    public ArrayList<String[]> solveNQueens(int n) {
        ArrayList<String[]> res = new ArrayList<String[]>();
        helper1(n, 0, new int[n], res);
        return res;
    }

    private void helper1(int n, int row, int[] columnForRow, ArrayList<String[]> res) {
        if (row == n) {
            String[] item = new String[n];
            for (int i = 0; i < n; i++) {
                StringBuilder strRow = new StringBuilder();
                for (int j = 0; j < n; j++) {
                    if (columnForRow[i] == j)
                        strRow.append('Q');
                    else
                        strRow.append('.');
                }
                item[i] = strRow.toString();
            }
            res.add(item);
            return;
        }
        for (int i = 0; i < n; i++) {
            columnForRow[row] = i;
            if (check(row, columnForRow)) {
                helper1(n, row + 1, columnForRow, res);
            }
        }
    }

    private boolean check(int row, int[] columnForRow) {
        for (int i = 0; i < row; i++) {
            if (columnForRow[row] == columnForRow[i] || Math.abs(columnForRow[row] - columnForRow[i]) == row - i)
                return false;
        }
        return true;
    }

    public int totalNQueens(int n) {
        ArrayList<Integer> res = new ArrayList<Integer>();
        res.add(0);
        helper2(n, 0, new int[n], res);
        return res.get(0);
    }

    private void helper2(int n, int row, int[] columnForRow, ArrayList<Integer> res) {
        if (row == n) {
            res.set(0, res.get(0) + 1);
            return;
        }
        for (int i = 0; i < n; i++) {
            columnForRow[row] = i;
            if (check(row, columnForRow)) {
                helper2(n, row + 1, columnForRow, res);
            }
        }
    }

    public double pow(double x, int n) {
        if (n == 0) return 1.0;
        double half = pow(x, n / 2);
        if (n % 2 == 0) {
            return half * half;
        } else if (n > 0) {
            return half * half * x;
        } else {
            return half / x * half;
        }
    }

    public ArrayList<String> anagrams(String[] strs) {
        ArrayList<String> result = new ArrayList<>();
        Map<String, ArrayList<String>> map = new HashMap<>();
        for (String s : strs) {
            String key = sortString(s);
            if (!map.containsKey(key)) {
                map.put(key, new ArrayList<String>());
            }
            map.get(key).add(s);
        }
        for (String s : map.keySet()) {
            ArrayList<String> list = map.get(s);
            if (list.size() > 1)
                result.addAll(list);
        }
        ArrayList<String> output = new ArrayList<>();
        for (int i = result.size() - 1; i >= 0; i--) {
            output.add(result.get(i));
        }
        return output;
    }

    private String sortString(String string) {
        char[] chars = string.toCharArray();
        Arrays.sort(chars);
        return new String(chars);
    }

    public void rotate(int[][] matrix) {
        int n = matrix.length;
        for (int i = 0; i < n / 2; i++) {
            for (int j = i; j < n - 1 - i; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - j - 1][i];
                matrix[n - j - 1][i] = matrix[n - i - 1][n - j - 1];
                matrix[n - i - 1][n - j - 1] = matrix[j][n - i - 1];
                matrix[j][n - i - 1] = temp;
            }
        }
    }

    public ArrayList<ArrayList<Integer>> permute(int[] num) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (num.length == 0) return result;
        permuteSwap(0, num, result);
        return result;
    }

    private void permuteSwap(int i, int[] num, ArrayList<ArrayList<Integer>> result) {
        ArrayList<Integer> list = new ArrayList<>();
        if (i == num.length) {
            for (int j = 0; j < num.length; j++) {
                list.add(num[j]);
            }
            result.add(list);
            return;
        } else {
            for (int j = i; j < num.length; j++) {
                int temp = num[i];
                num[i] = num[j];
                num[j] = temp;
                permuteSwap(i + 1, num, result);
                temp = num[i];
                num[i] = num[j];
                num[j] = temp;
            }
        }
    }

    public ArrayList<ArrayList<Integer>> permuteUnique(int[] num) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (num.length == 0) return result;
        permuteSwap2(0, num.length - 1, num, result);
        return result;
    }

    private void permuteSwap2(int start, int end, int[] num, ArrayList<ArrayList<Integer>> result) {
        if (start == end) {
            ArrayList<Integer> list = new ArrayList<>();
            for (int i = 0; i < num.length; i++) {
                list.add(num[i]);
            }
            result.add(list);
            return;
        } else {
            for (int i = start; i <= end; i++) {
                if (!findSame(num, start, i)) {
                    int temp = num[start];
                    num[start] = num[i];
                    num[i] = temp;
                    permuteSwap2(start + 1, end, num, result);
                    temp = num[start];
                    num[start] = num[i];
                    num[i] = temp;
                }
            }
        }
    }

    private boolean findSame(int[] num, int start, int end) {
        for (int i = start; i < end; i++) {
            if (num[i] == num[end]) {
                return true;
            }
        }
        return false;
    }

    public boolean isMatch(String s, String p) {
        char[] sCharArray = s.toCharArray();
        char[] pCharArray = p.toCharArray();
        boolean[][] dp = new boolean[256][256];
        int l = 0;
        if (p.length() != 0) {
            for (int i = 0; i < p.length(); i++) {
                if (pCharArray[i] != '*') l++;
            }
        }
        if (l > s.length()) return false;//p的字符数加上'?'的数目要小于s的字符数，否则根本不能匹配
        dp[0][0] = true;
        for (int i = 1; i <= p.length(); i++) {
            if (dp[0][i - 1] && pCharArray[i - 1] == '*') dp[0][i] = true;
            for (int j = 1; j <= s.length(); j++) {
                if (pCharArray[i - 1] == '*') dp[j][i] = dp[j][i - 1] || dp[j - 1][i];
                else if (pCharArray[i - 1] == '?' || pCharArray[i - 1] == sCharArray[j - 1])
                    dp[j][i] = dp[j - 1][i - 1];
                else dp[j][i] = false;
            }
        }
        return dp[s.length()][p.length()];
    }

    public String multiply(String num1, String num2) {
        BigDecimal n1 = new BigDecimal(num1);
        BigDecimal n2 = new BigDecimal(num2);
        return n1.multiply(n2).toString();
    }

    public int trap(int[] A) {
        int n = A.length;
        if (n <= 2) return 0;
        if (n == 3) {
            int sum = Math.min(A[0] - A[1], A[2] - A[1]);
            if (sum > 0) return sum;
            else return 0;
        }
        int[] left = new int[n];
        int[] right = new int[n];
        int[] sum = new int[n];
        for (int i = 1; i < n - 1; i++) {
            left[i] = Math.max(left[i - 1], A[i - 1]);
        }
        for (int i = n - 2; i > 0; i--) {
            right[i] = Math.max(right[i + 1], A[i + 1]);
        }
        for (int i = 1; i < n - 1; i++) {
            sum[i] = Math.min(left[i], right[i]) - A[i];
        }
        int result = 0;
        for (int i = 1; i < n - 1; i++) {
            if (sum[i] > 0) {
                result = result + sum[i];
            }
        }
        return result;
    }

    public int firstMissingPositive(int[] A) {
        int n = A.length;
        if (n == 1) {
            if (A[0] <= 0) return 1;
            if (A[0] == 1) return 2;
            else return 1;
        }
        for (int i = 1; i < n; i++) {
            while (A[i] > 0 && A[i] <= n && A[A[i] - 1] != A[i]) {
                int temp = A[A[i] - 1];
                A[A[i] - 1] = A[i];
                A[i] = temp;
            }
        }
        for (int i = 0; i < n; i++) {
            if (A[i] != i + 1) return i + 1;
        }
        return A.length + 1;
    }

    public ArrayList<ArrayList<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        ArrayList<Integer> list = new ArrayList<>();
        backTrackingSum(candidates, 0, target, list, result);
        return result;
    }

    private void backTrackingSum(int[] cadidates, int start, int target, ArrayList<Integer> list,
                                 ArrayList<ArrayList<Integer>> result) {
        if (target == 0) {
            result.add(new ArrayList<Integer>(list));
            return;
        } else {
            for (int i = start; i < cadidates.length && cadidates[i] <= target; i++) {
                list.add(cadidates[i]);
                backTrackingSum(cadidates, i, target - cadidates[i], list, result);
                list.remove(list.size() - 1);
            }
        }
    }

    public ArrayList<ArrayList<Integer>> combinationSum2(int[] num, int target) {
        Arrays.sort(num);
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        ArrayList<Integer> list = new ArrayList<>();
        backTrackingSum2(num, 0, target, list, result);
        return result;
    }

    private void backTrackingSum2(int[] num, int start, int target, ArrayList<Integer> list,
                                  ArrayList<ArrayList<Integer>> result) {
        if (target == 0) {
            boolean isExist = false;
            for (int i = result.size() - 1; i >= 0; i--) {
                ArrayList<Integer> exist = result.get(i);
                if (exist.equals(list)) {
                    isExist = true;
                    break;
                }
            }
            if (isExist == false) {
                result.add(new ArrayList<Integer>(list));
            }
            return;
        } else {
            for (int i = start; i < num.length && num[i] <= target; i++) {
                list.add(num[i]);
                backTrackingSum2(num, i + 1, target - num[i], list, result);
                list.remove(list.size() - 1);
            }
        }
    }

    public String countAndSay(int n) {
        String result = "1";
        for (int i = 1; i < n; i++) {
            result = countString(result);
        }
        return result;
    }

    private String countString(String result) {
        char c = result.charAt(0);
        int count = 1;
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 1; i < result.length(); i++) {
            if (result.charAt(i) == c) {
                count++;
                continue;
            } else {
                stringBuilder.append(String.valueOf(count) + c);
                c = result.charAt(i);
                count = 1;
            }
        }
        stringBuilder.append(String.valueOf(count) + c);
        return stringBuilder.toString();
    }

    public void solveSudoku(char[][] board) {
        if (board == null || board.length != 9 || board[0].length != 9)
            return;
        sudokuHelper(board, 0, 0);
    }

    private boolean sudokuHelper(char[][] board, int i, int j) {
        if (j >= 9)
            return sudokuHelper(board, i + 1, 0);
        if (i == 9) {
            return true;
        }
        if (board[i][j] == '.') {
            for (int k = 1; k <= 9; k++) {
                board[i][j] = (char) (k + '0');
                if (isValid(board, i, j)) {
                    if (sudokuHelper(board, i, j + 1))
                        return true;
                }
                board[i][j] = '.';
            }
        } else {
            return sudokuHelper(board, i, j + 1);
        }
        return false;
    }

    private boolean isValid(char[][] board, int i, int j) {
        for (int k = 0; k < 9; k++) {
            if (k != j && board[i][k] == board[i][j])
                return false;
        }
        for (int k = 0; k < 9; k++) {
            if (k != i && board[k][j] == board[i][j])
                return false;
        }
        for (int row = i / 3 * 3; row < i / 3 * 3 + 3; row++) {
            for (int col = j / 3 * 3; col < j / 3 * 3 + 3; col++) {
                if ((row != i || col != j) && board[row][col] == board[i][j])
                    return false;
            }
        }
        return true;
    }

    public boolean isValidSudoku(char[][] board) {
        if (board == null || board.length != 9 || board[0].length != 9)
            return false;
        for (int i = 0; i < 9; i++) {
            boolean[] map = new boolean[9];
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.') {
                    if (map[(int) (board[i][j] - '1')]) {
                        return false;
                    }
                    map[(int) (board[i][j] - '1')] = true;
                }
            }
        }
        for (int j = 0; j < 9; j++) {
            boolean[] map = new boolean[9];
            for (int i = 0; i < 9; i++) {
                if (board[i][j] != '.') {
                    if (map[(int) (board[i][j] - '1')]) {
                        return false;
                    }
                    map[(int) (board[i][j] - '1')] = true;
                }
            }
        }
        for (int block = 0; block < 9; block++) {
            boolean[] map = new boolean[9];
            for (int i = block / 3 * 3; i < block / 3 * 3 + 3; i++) {
                for (int j = block % 3 * 3; j < block % 3 * 3 + 3; j++) {
                    if (board[i][j] != '.') {
                        if (map[(int) (board[i][j] - '1')]) {
                            return false;
                        }
                        map[(int) (board[i][j] - '1')] = true;
                    }
                }
            }
        }
        return true;
    }

    public int searchInsert(int[] A, int target) {
        for (int i = 0; i < A.length; i++) {
            if (A[i] >= target) return i;
        }
        return A.length;
    }

    public int[] searchRange(int[] A, int target) {
        if (A.length == 1) {
            if (A[0] == target) return new int[]{0, 0};
            else return new int[]{-1, -1};
        }
        int left = 0, right = A.length - 1;
        while (left < right) {
            if (A[left] != target) left++;
            if (A[right] != target) right--;
            if (A[left] == target && A[right] == target) {
                return new int[]{left, right};
            }
        }
        return new int[]{-1, -1};
    }

    public int longestValidParentheses(String s) {
        if (s.length() == 0) return 0;
        int len = 0, last = -1;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') stack.push(i);
            else {
                if (stack.isEmpty()) last = i;
                else {
                    stack.pop();
                    if (stack.isEmpty()) len = Math.max(len, i - last);
                    else len = Math.max(len, i - stack.peek());
                }
            }
        }
        return len;
    }

    public void nextPermutation(int[] num) {
        if (num == null || num.length == 0)
            return;
        int i = num.length - 2;
        while (i >= 0 && num[i] >= num[i + 1]) {
            i--;
        }
        if (i >= 0) {
            int j = i + 1;
            while (j < num.length && num[j] > num[i]) {
                j++;
            }
            j--;
            int temp = num[i];
            num[i] = num[j];
            num[j] = temp;
        }
        int l = i + 1;
        int r = num.length - 1;
        while (l < r) {
            int temp = num[l];
            num[l] = num[r];
            num[r] = temp;
            l++;
            r--;
        }
    }

    public ArrayList<Integer> findSubstring(String S, String[] L) {
        ArrayList<Integer> res = new ArrayList<Integer>();
        if (S == null || S.length() == 0 || L == null || L.length == 0)
            return res;
        HashMap<String, Integer> map = new HashMap<String, Integer>();
        for (int i = 0; i < L.length; i++) {
            if (map.containsKey(L[i])) {
                map.put(L[i], map.get(L[i]) + 1);
            } else {
                map.put(L[i], 1);
            }
        }
        for (int i = 0; i < L[0].length(); i++) {
            HashMap<String, Integer> curMap = new HashMap<String, Integer>();
            int count = 0;
            int left = i;
            for (int j = i; j <= S.length() - L[0].length(); j += L[0].length()) {
                String str = S.substring(j, j + L[0].length());

                if (map.containsKey(str)) {
                    if (curMap.containsKey(str))
                        curMap.put(str, curMap.get(str) + 1);
                    else
                        curMap.put(str, 1);
                    if (curMap.get(str) <= map.get(str))
                        count++;
                    else {
                        while (curMap.get(str) > map.get(str)) {
                            String temp = S.substring(left, left + L[0].length());
                            if (curMap.containsKey(temp)) {
                                curMap.put(temp, curMap.get(temp) - 1);
                                if (curMap.get(temp) < map.get(temp))
                                    count--;
                            }
                            left += L[0].length();
                        }
                    }
                    if (count == L.length) {
                        res.add(left);
                        String temp = S.substring(left, left + L[0].length());
                        if (curMap.containsKey(temp))
                            curMap.put(temp, curMap.get(temp) - 1);
                        count--;
                        left += L[0].length();
                    }
                } else {
                    curMap.clear();
                    count = 0;
                    left = j + L[0].length();
                }
            }
        }
        Collections.sort(res);
        return res;
    }

    public int divide(int dividend, int divisor) {
        int sign = 1;
        if (dividend < 0) sign = -sign;
        if (divisor < 0) sign = -sign;
        long temp = Math.abs((long) dividend);
        long temp2 = Math.abs((long) divisor);
        long c = 1;
        while (temp > temp2) {
            temp2 = temp2 << 1;
            c = c << 1;
        }
        int res = 0;
        while (temp >= Math.abs((long) divisor)) {
            while (temp >= temp2) {
                temp -= temp2;
                res += c;
            }
            temp2 = temp2 >> 1;
            c = c >> 1;
        }
        if (sign > 0) return res;
        else return -res;
    }

    public String strStr(String haystack, String needle) {
        if (needle.length() == 0) return haystack;
        for (int i = 0; i < haystack.length(); i++) {
            if (haystack.length() - i + 1 < needle.length()) return null;
            int k = i;
            int j = 0;
            while (j < needle.length() && k < haystack.length() && needle.charAt(j) == haystack.charAt(k)) {
                j++;
                k++;
                if (j == needle.length()) return haystack.substring(i);
            }
        }
        return null;
    }

    public int removeElement(int[] A, int elem) {
        int index = 0;
        for (int i = 0; i < A.length; i++) {
            if (A[i] != elem) {
                A[index] = A[i];
                index++;
            }
        }
        return index;
    }

    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || k < 2) return head;
        ListNode first = new ListNode(-1);
        first.next = head;
        ListNode pre = first;
        ListNode cur = head;
        ListNode temp = null;
        int len = 0;
        while (head != null) {
            head = head.next;
            len++;
        }
        int count = len / k;
        while (count-- > 0) {
            int c = k;
            while (c-- > 1) {
                temp = cur.next;
                cur.next = temp.next;
                temp.next = pre.next;
                pre.next = temp;
            }
            pre = cur;
            cur = cur.next;
        }
        return first.next;
    }

    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode preHead = new ListNode(0);
        preHead.next = head;
        ListNode cur = preHead;
        while (cur.next != null && cur.next.next != null) {
            cur.next = swapListNode(cur.next, cur.next.next);
            cur = cur.next.next;
        }
        return preHead.next;
    }

    private ListNode swapListNode(ListNode node1, ListNode node2) {
        node1.next = node2.next;
        node2.next = node1;
        return node2;
    }

    public ListNode mergeKLists(ArrayList<ListNode> lists) {
        if (lists.size() == 0) return null;
        ListNode head = lists.get(0);
        for (int i = 1; i < lists.size(); i++) {
            head = mergeTwoLists(head, lists.get(i));
        }
        return head;
    }

    public ArrayList<String> generateParenthesis(int n) {
        ArrayList<String> result = new ArrayList<>();
        printParenthesis(n, 0, 0, "", result);
        return result;
    }

    private void printParenthesis(int n, int left, int right, String s, ArrayList<String> result) {
        if (right == n) result.add(s);
        if (left < n) printParenthesis(n, left + 1, right, s + "(", result);
        if (left > right) printParenthesis(n, left, right + 1, s + ")", result);
    }

    public boolean isValid(String s) {
        if (s.length() == 0) return true;
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(' || s.charAt(i) == '{' || s.charAt(i) == '[') {
                stack.push(s.charAt(i));
            } else {
                if (stack.isEmpty() || (s.charAt(i) == ')' && stack.pop() != '(')
                        || (s.charAt(i) == '}' && stack.pop() != '{')
                        || (s.charAt(i) == ']' && stack.pop() != '[')) {
                    return false;
                }
            }
        }
        if (stack.isEmpty()) return true;
        else return false;
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        if (head == null) return head;
        ListNode preHead = new ListNode(0);
        preHead.next = head;
        ListNode slow, fast;
        slow = fast = head;
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        if (fast == null) return slow.next;
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return preHead.next;
    }

    public ArrayList<String> letterCombinations(String digits) {
        ArrayList<String> result = new ArrayList<>();
        HashMap<Character, char[]> hashMap = new HashMap<>();
        hashMap.put('0', new char[]{' '});
        hashMap.put('2', new char[]{'a', 'b', 'c'});
        hashMap.put('3', new char[]{'d', 'e', 'f'});
        hashMap.put('4', new char[]{'g', 'h', 'i'});
        hashMap.put('5', new char[]{'j', 'k', 'l'});
        hashMap.put('6', new char[]{'m', 'n', 'o'});
        hashMap.put('7', new char[]{'p', 'q', 'r', 's'});
        hashMap.put('8', new char[]{'t', 'u', 'v'});
        hashMap.put('9', new char[]{'w', 'x', 'y', 'z'});
        getString(digits, 0, result, "", hashMap);
        return result;
    }

    private void getString(String digits, int position, ArrayList<String> result,
                           String str, HashMap<Character, char[]> hashMap) {
        if (position < digits.length()) {
            if (hashMap.containsKey(digits.charAt(position))) {
                for (char c : hashMap.get(digits.charAt(position))) {
                    String newStr = str + c;
                    getString(digits, position + 1, result, newStr, hashMap);
                }
            }
        } else result.add(str);
    }

    public int[] twoSum(int[] numbers, int target) {
        int[] result = new int[2];
        HashMap<Integer, Integer> hashMap = new HashMap<>();
        for (int i = 0; i < numbers.length; i++) {
            if (hashMap.containsKey(numbers[i])) {
                result[0] = hashMap.get(numbers[i]) + 1;
                result[1] = i + 1;
                break;
            } else hashMap.put(target - numbers[i], i);
        }
        return result;
    }

    public ArrayList<ArrayList<Integer>> threeSum(int[] num) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (num.length < 3) return result;
        Arrays.sort(num);
        for (int i = 0; i < num.length - 2; i++) {
            if (i == 0 || (i > 0 && num[i] != num[i - 1])) {
                int left = i + 1, right = num.length - 1, sum = 0 - num[i];
                while (left < right) {
                    if (num[left] + num[right] == sum) {
                        ArrayList<Integer> list = new ArrayList<>();
                        list.add(num[i]);
                        list.add(num[left]);
                        list.add(num[right]);
                        result.add(list);
                        left++;
                        right--;
                        while (left < right && num[left] == num[left - 1]) left++;
                        while (right > left && num[right] == num[right + 1]) right--;
                    } else if (num[left] + num[right] > sum) right--;
                    else left++;
                }
            }
        }
        return result;
    }

    public int threeSumClosest(int[] num, int target) {
        int sum, error, result, min = Integer.MAX_VALUE;
        sum = error = result = 0;
        for (int i = 0; i < num.length; i++) {
            for (int j = i + 1; j < num.length; j++) {
                for (int k = j + 1; k < num.length; k++) {
                    sum = num[i] + num[j] + num[k];
                    error = Math.abs(target - sum);
                    if (error < min) {
                        min = error;
                        result = sum;
                    }
                }
            }
        }
        return result;
    }

    public ArrayList<ArrayList<Integer>> fourSum(int[] num, int target) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (num.length < 4) return result;
        Arrays.sort(num);
        for (int i = 0; i < num.length - 3; i++) {
            for (int j = i + 1; j < num.length - 2; j++) {
                int left = j + 1;
                int right = num.length - 1;
                while (left < right) {
                    int sum = num[left] + num[right];
                    if (sum < target - num[i] - num[j]) left++;
                    if (sum > target - num[i] - num[j]) right--;
                    if (sum == (target - num[i] - num[j])) {
                        ArrayList<Integer> list = new ArrayList<>();
                        list.add(num[i]);
                        list.add(num[j]);
                        list.add(num[left]);
                        list.add(num[right]);
                        result.add(list);
                        int temp1 = num[left];
                        int temp2 = num[right];
                        while (left < right && num[left] == temp1) left++;
                        while (left < right && num[right] == temp2) right--;
                    }
                }
                while (j + 1 < num.length - 2 && num[j + 1] == num[j]) j++;
            }
            while (i + 1 < num.length - 3 && num[i + 1] == num[i]) i++;
        }
        return result;
    }

    public String longestCommonPrefix(String[] strs) {
        if (strs.length == 0) return "";
        String result = strs[0];
        for (int i = 1; i < strs.length; i++) {
            int j = 0;
            String preFix = "";
            while (j < result.length() && j < strs[i].length() && result.charAt(j) == strs[i].charAt(j)) {
                preFix = preFix + strs[i].charAt(j);
                j++;
            }
            result = preFix;
        }
        return result;
    }

    public int romanToInt(String s) {
        HashMap<Character, Integer> hashMap = new HashMap<>();
        hashMap.put('I', 1);
        hashMap.put('V', 5);
        hashMap.put('X', 10);
        hashMap.put('L', 50);
        hashMap.put('C', 100);
        hashMap.put('D', 500);
        hashMap.put('M', 1000);
        int result = 0, preValue = 0;
        for (int i = s.length() - 1; i >= 0; i--) {
            int curValue = hashMap.get(s.charAt(i));
            if (curValue < preValue) result = result - curValue;
            else result = result + curValue;
            preValue = curValue;
        }
        return result;
    }

    public String intToRoman(int num) {
        String M[] = {"", "M", "MM", "MMM"};
        String C[] = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
        String X[] = {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
        String I[] = {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};
        return M[num / 1000] + C[(num % 1000) / 100] + X[(num % 100) / 10] + I[num % 10];
    }

    public int maxArea(int[] height) {
        int result = 0;
        if (height.length < 2) return 0;
        int left = 0, right = height.length - 1;
        while (left < right) {
            int area = (right - left) * Math.min(height[left], height[right]);
            if (area > result) result = area;
            if (height[left] > height[right]) right--;
            else left++;
        }
        return result;
    }

    public boolean isPalindrome(int x) {
        if (x < 0)
            return false;
        int div = 1;
        while (div <= x / 10)
            div *= 10;
        while (x > 0) {
            if (x / div != x % 10)
                return false;
            x = (x % div) / 10;
            div /= 100;
        }
        return true;
    }

    public int atoi(String str) {
        str = str.trim();
        if (str.length() == 0) return 0;
        String result = "";
        int inx = 0;
        String minus = "";
        if (str.charAt(0) == '-') {
            minus = "-";
            inx++;
        } else if (str.charAt(0) == '+') {
            minus = "+";
            inx++;
        }
        for (int i = inx; i < str.length(); i++) {
            if (str.charAt(i) >= '0' && str.charAt(i) <= '9') result = result + str.charAt(i);
            else break;
        }
        if (result == "") return 0;
        if (Long.valueOf(minus + result) > Integer.MAX_VALUE) return Integer.MAX_VALUE;
        if (Long.valueOf(minus + result) < Integer.MIN_VALUE) return Integer.MIN_VALUE;
        return Integer.valueOf(minus + result);
    }

    public int reverse(int x) {
        if (x == 0) return 0;
        String num = String.valueOf(x);
        int inx = 0;
        boolean minus = false;
        if (num.charAt(0) == '-') {
            minus = true;
            inx++;
        }
        long result = 0;
        int flag = 1;
        if (minus) flag = -1;
        for (int i = num.length() - 1; i >= inx; i--) {
            result = result * 10 + flag * (num.charAt(i) - '0');
            if (result > Integer.MAX_VALUE || result < Integer.MIN_VALUE) return 0;
        }
        return (int) result;
    }

    public String convert(String s, int nRows) {
        if (s.length() < 0 || s.length() < nRows) return s;
        StringBuilder[] stringBuilders = new StringBuilder[nRows];
        for (int i = 0; i < nRows; i++) {
            stringBuilders[i] = new StringBuilder();
        }
        char[] c = s.toCharArray();
        int index = 0;
        while (index < s.length()) {
            for (int i = 0; i < nRows && index < s.length(); i++) {
                stringBuilders[i].append(c[index]);
                index++;
            }
            for (int i = nRows - 2; i > 0 && index < s.length(); i--) {
                stringBuilders[i].append(c[index]);
                index++;
            }
        }
        for (int i = 1; i < nRows; i++) {
            stringBuilders[0].append(stringBuilders[i]);
        }
        return stringBuilders[0].toString();
    }

    public String longestPalindrome(String s) {
        if (s.length() < 2) return s;
        for (int i = 0; i < s.length() - 1; i++) {
            findLongestPalindrome(s, i, i);
            findLongestPalindrome(s, i, i + 1);
        }
        return s.substring(startInLongestPalindrome, startInLongestPalindrome + maxLenInLongestPalindrome);
    }

    private void findLongestPalindrome(String s, int i, int j) {
        while (i >= 0 && j < s.length() && s.charAt(i) == s.charAt(j)) {
            i--;
            j++;
        }
        if (maxLenInLongestPalindrome < j - i - 1) {
            startInLongestPalindrome = i + 1;
            maxLenInLongestPalindrome = j - i - 1;
        }
    }

    private int startInLongestPalindrome = 0, maxLenInLongestPalindrome = 0;

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        if (l2 == null) return l1;
        ListNode head = new ListNode(0);
        ListNode cur = head;
        int temp = 0;
        while (l1 != null || l2 != null || temp != 0) {
            if (l1 != null) {
                temp = temp + l1.val;
                l1 = l1.next;
            }
            if (l2 != null) {
                temp = temp + l2.val;
                l2 = l2.next;
            }
            cur.next = new ListNode(temp % 10);
            cur = cur.next;
            temp = temp / 10;
        }
        return head.next;
    }

    public int lengthOfLongestSubstring(String s) {
        if (s.length() == 0) return 0;
        HashMap<Character, Integer> hashMap = new HashMap<>();
        int leftBound = 0;
        int max = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            int isSame = 0;
            if (hashMap.containsKey(c)) isSame = hashMap.get(c) + 1;
            leftBound = Math.max(leftBound, isSame);
            max = Math.max(max, i - leftBound + 1);
            hashMap.put(c, i);
        }
        return max;
    }

    public double findMedianSortedArrays(int A[], int B[]) {
        double median_fir = 0;
        int m = A.length, n = B.length;
        if (A == null || B == null || m + n == 0) {
            return 0;
        }
        int indexA = 0, indexB = 0;
        while (indexA + indexB != (m + n + 1) / 2) {
            int a = (indexA == m) ? Integer.MAX_VALUE : A[indexA];
            int b = (indexB == n) ? Integer.MAX_VALUE : B[indexB];
            if (a < b) {
                median_fir = a;
                indexA++;
            } else {
                median_fir = b;
                indexB++;
            }
        }
        if ((m + n) % 2 == 1)
            return median_fir;
        else {
            int temp_a = (indexA == m) ? Integer.MAX_VALUE : A[indexA];
            int temp_b = (indexB == n) ? Integer.MAX_VALUE : B[indexB];
            double median_sec = (temp_a < temp_b) ? temp_a : temp_b;
            return (median_fir + median_sec) / 2;
        }
    }

    
}