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
        if (end <= 0) {
            String ans = tmp.get(tmp.size() - 1);
            for (int i = tmp.size() - 2; i >= 0; i--) {
                ans = ans + (" " + tmp.get(i));
            }
            result.add(ans);
            return;
        }
        for (String str : dp[end]) {
            tmp.add(str);
            dfsSearch(dp, end - str.length(), result, tmp);
            tmp.remove(tmp.size() - 1);
        }
    }

    private class RandomListNode {
        int label;
        RandomListNode next, random;

        RandomListNode(int x) {
            this.label = x;
        }
    }

    public RandomListNode copyRandomList(RandomListNode head) {
        if (head == null) return head;
        RandomListNode newHead = new RandomListNode(head.label);
        RandomListNode oldp = head.next;
        RandomListNode newp = newHead;
        Map<RandomListNode, RandomListNode> map = new HashMap<RandomListNode, RandomListNode>();
        //采用map结构来存储对应的关系
        map.put(newp, head);
        while (oldp != null) {//复制旧的链表
            RandomListNode newTemp = new RandomListNode(oldp.label);
            map.put(newTemp, oldp);
            newp.next = newTemp;
            newp = newp.next;
            oldp = oldp.next;
        }
        oldp = head;
        newp = newHead;
        while (newp != null) {//复制random指针
            newp.random = map.get(newp).random;//取得旧节点的random指针
            newp = newp.next;
            oldp = oldp.next;
        }
        return head;
    }

    public int singleNumber(int[] A) {
        int result = 0;
        for (int i = 0; i < A.length; i++) {
            result = result ^ A[i];
        }
        return result;
    }

    public int singleNumber2(int[] A) {
        Arrays.sort(A);
        for (int i = 0; i < A.length - 3; i = i + 3) {
            if (A[i] != A[i + 1] && A[i + 1] == A[i + 2]) {
                return A[i];
            }
        }
        return A[A.length - 1];
    }

    public int candy(int[] ratings) {
        int n = ratings.length;
        int[] num = new int[n];
        Arrays.fill(num, 1);//保证每人都有一颗
        for (int i = 1; i < n; i++) {//从左往右扫一遍，保证左邻居颗数维持题意
            if (ratings[i] > ratings[i - 1] && num[i] <= num[i - 1]) {
                num[i] = num[i - 1] + 1;
            }
        }
        for (int i = n - 1; i > 0; i--) {//从右往左扫一遍，保证右邻居颗数维持题意
            if (ratings[i] < ratings[i - 1] && num[i] >= num[i - 1]) {
                num[i - 1] = num[i] + 1;
            }
        }
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum = sum + num[i];
        }
        return sum;
    }

    public int canCompleteCircuit(int[] gas, int[] cost) {
        int end = gas.length - 1;
        int start = 0;
        int sum = gas[end] - cost[end];
        while (end > start) {
            if (sum >= 0) {
                sum = sum + gas[start] - cost[start];
                start++;
            } else {
                end--;
                sum = sum + gas[end] - cost[end];
            }
        }
        if (sum >= 0) return end;
        else return -1;
    }

    public ArrayList<ArrayList<String>> partition(String s) {
        ArrayList<ArrayList<String>> result = new ArrayList<ArrayList<String>>();
        ArrayList<String> list = new ArrayList<String>();//存储当前一种切割的结果
        if (s.length() == 0) return result;
        getPartition(result, list, s);
        return result;
    }

    private void getPartition(ArrayList<ArrayList<String>> result, ArrayList<String> list, String string) {
        if (string.length() == 0) result.add(new ArrayList<String>(list));
        //如果string长度为0了，说明这种切分满足都是回文，将这种切分的结果list加入到result中
        int len = string.length();
        for (int i = 1; i <= len; i++) {
            String str = string.substring(0, i);
            if (isPartition(str)) {
                list.add(str);
                getPartition(result, list, string.substring(i));//传入i之后的继续切割
                list.remove(list.size() - 1);//不加这句通不过AC，因为最后一个会是string本身
            }
        }
    }

    private boolean isPartition(String string) {//判断是否是回文数
        int i = 0;
        int j = string.length() - 1;
        while (i < j) {
            if (string.charAt(i) != string.charAt(j)) return false;
            i++;
            j--;
        }
        return true;
    }

    public int minCut(String s) {
        int len = s.length();
        int[] cut = new int[len + 1];
        boolean[][] c = new boolean[len + 1][len + 1];
        if (s.length() == 0) return 0;
        for (int i = 0; i < len; i++) {
            cut[i] = len - i;
        }
        for (int i = len - 1; i >= 0; i--) {
            for (int j = i; j < len; j++) {
                if (s.charAt(i) == s.charAt(j) && (j - i < 2)) {
                    c[i][j] = true;
                    cut[i] = Math.min(cut[i], cut[j + 1] + 1);
                } else if (s.charAt(i) == s.charAt(j) && c[i + 1][j - 1]) {
                    c[i][j] = true;
                    cut[i] = Math.min(cut[i], cut[j + 1] + 1);
                }
            }
        }
        return cut[0] - 1;
    }

    public void solve(char[][] board) {
        if (board.length == 0) return;
        int len = board[0].length;
        boolean[][] visited = new boolean[board.length][len];
        LinkedList<Integer> queue = new LinkedList<>();//用队列存储坐标
        int[][] direction = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};//定义四个方向
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (board[i][j] == 'O' && visited[i][j] == false) {
                    boolean surrounded = true;
                    ArrayList<Integer> visitedPoint = new ArrayList<>();//存储已经访问过的点
                    queue.offer(i * board[0].length + j);
                    visited[i][j] = true;
                    while (queue.isEmpty() == false) {//BFS搜索
                        int point = queue.poll();
                        visitedPoint.add(point);
                        int x = point / len;//相除和取模拿到x，y坐标
                        int y = point % len;
                        for (int k = 0; k < 4; k++) {//分别向四个方向进行搜索
                            int nextX = x + direction[k][0];
                            int nextY = y + direction[k][1];
                            if (nextX >= 0 && nextX < board.length && nextY >= 0 && nextY < len) {
                                if (board[nextX][nextY] == 'O' && visited[nextX][nextY] == false) {
                                    queue.offer(nextX * len + nextY);
                                    visited[nextX][nextY] = true;
                                }
                            } else {
                                surrounded = false;//边缘点不用考虑
                            }
                        }
                    }
                    if (surrounded) {
                        for (int p : visitedPoint) {
                            board[p / len][p % len] = 'X';
                        }
                    }
                }
            }
        }
    }

    public int sumNumbers(TreeNode root) {
        int sum = 0;
        if (root == null) return 0;
        else return preOrderSumNumbers(root, sum);
    }

    private int preOrderSumNumbers(TreeNode root, int sum) {
        if (root == null) return 0;
        sum = sum * 10 + root.val;
        if (root.left == null && root.right == null) return sum;
        else return preOrderSumNumbers(root.left, sum) + preOrderSumNumbers(root.right, sum);
    }

    public int longestConsecutive(int[] num) {
        int n = num.length;
        int result = 1;
        HashSet<Integer> set = new HashSet<>();
        for (int i = 0; i < n; i++) {
            set.add(num[i]);
        }
        for (int i = 0; i < n; i++) {
            int pre = num[i] - 1;
            int next = num[i] + 1;
            int count = 1;
            while (set.remove(pre)) {
                pre--;
                count++;
            }
            while (set.remove(next)) {
                next++;
                count++;
            }
            if (count > result) result = count;
        }
        return result;
    }

    class UndirectedGraphNode {
        int label;
        ArrayList<UndirectedGraphNode> neighbors;

        UndirectedGraphNode(int x) {
            label = x;
            neighbors = new ArrayList<UndirectedGraphNode>();
        }
    }

    public UndirectedGraphNode cloneGraphBFS(UndirectedGraphNode node) {//BFS
        if (node == null) return null;
        HashMap<UndirectedGraphNode, UndirectedGraphNode> hashMap = new HashMap<>();//哈希表用来存储已经访问过的节点
        LinkedList<UndirectedGraphNode> queue = new LinkedList<>();
        UndirectedGraphNode head = new UndirectedGraphNode(node.label);//新的表头
        hashMap.put(node, head);
        queue.offer(node);
        while (queue.isEmpty() == false) {
            UndirectedGraphNode curNode = queue.poll();
            for (UndirectedGraphNode neighbor : curNode.neighbors) {
                if (hashMap.containsKey(neighbor) == false) {//哈希表没有访问过
                    queue.offer(neighbor);
                    UndirectedGraphNode newNeighbor = new UndirectedGraphNode(neighbor.label);
                    hashMap.put(neighbor, newNeighbor);//说明该点已经被访问了
                }
                hashMap.get(curNode).neighbors.add(hashMap.get(neighbor));//拷贝到head里
            }
        }
        return head;
    }

    public UndirectedGraphNode cloneGraphDFS(UndirectedGraphNode node) {//DFS
        if (node == null) return null;
        HashMap<UndirectedGraphNode, UndirectedGraphNode> hashMap = new HashMap<>();
        UndirectedGraphNode head = new UndirectedGraphNode(node.label);
        hashMap.put(node, head);
        DFSSearch(hashMap, node);
        return head;
    }

    private void DFSSearch(HashMap<UndirectedGraphNode, UndirectedGraphNode> hashMap, UndirectedGraphNode node) {
        if (node == null) return;
        for (UndirectedGraphNode neighbor : node.neighbors) {
            if (hashMap.containsKey(neighbor) == false) {
                UndirectedGraphNode newNeighbor = new UndirectedGraphNode(neighbor.label);
                hashMap.put(neighbor, newNeighbor);
                DFSSearch(hashMap, neighbor);
            }
            hashMap.get(node).neighbors.add(hashMap.get(neighbor));
        }
    }

    public int ladderLength(String start, String end, HashSet<String> dict) {
        if (start.equals(end)) return 0;
        LinkedList<String> queue = new LinkedList<>();
        HashMap<String, Integer> dist = new HashMap<>();//保存变化的中间结果，变化一位之后的单词和从start变化过来的步数
        queue.offer(start);
        dist.put(start, 1);
        while (queue.isEmpty() == false) {
            String head = queue.poll();
            int headDist = dist.get(head);
            for (int i = 0; i < head.length(); i++) {
                for (char ch = 'a'; ch < 'z'; ch++) {
                    if (head.charAt(i) == ch) continue;
                    String newString = head;
                    StringBuilder stringBuilder = new StringBuilder(head);
                    stringBuilder.setCharAt(i, ch);
                    if (stringBuilder.toString().equals(end)) return headDist + 1;
                    if (dict.contains(stringBuilder.toString()) && dist.containsKey(stringBuilder.toString()) == false) {
                        queue.add(stringBuilder.toString());
                        dist.put(stringBuilder.toString(), headDist + 1);
                    }
                }
            }
        }
        return 0;
    }

    public boolean isPalindrome(String s) {
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            if (Character.isDigit(s.charAt(i)) || Character.isLetter(s.charAt(i))) {
                stringBuilder.append(s.charAt(i));
            }
        }//转化成只剩下字母和数字的字符串
        String str = new String(stringBuilder.toString());
        int i = 0;
        int j = str.length() - 1;
        while (i < j) {
            if (Character.toLowerCase(str.charAt(i)) != Character.toLowerCase(str.charAt(j))) return false;
            i++;
            j--;
        }
        return true;
    }

    public int maxPathSum(TreeNode root) {
        if (root == null) return 0;
        ArrayList<Integer> result = new ArrayList<>();
        result.add(Integer.MIN_VALUE);
        getMaxPathSum(root, result);
        return result.get(0);
    }

    private int getMaxPathSum(TreeNode node, ArrayList<Integer> result) {
        if (node == null) return 0;
        int left = Math.max(0, getMaxPathSum(node.left, result));
        int right = Math.max(0, getMaxPathSum(node.right, result));
        result.set(0, Math.max(result.get(0), node.val + left + right));
        return Math.max(left, right) + node.val;
    }

    public int maxProfit(int[] prices) {
        if (prices.length == 0) return 0;
        int profit = 0;
        int min = prices[0];
        for (int i = 0; i < prices.length; i++) {
            min = Math.min(min, prices[i]);
            profit = Math.max(profit, prices[i] - min);
        }
        return profit;
    }

    public int maxProfit2(int[] prices) {
        if (prices.length == 0) return 0;
        int[] dayProfit = new int[prices.length];
        Arrays.fill(dayProfit, 0);
        for (int i = 1; i < prices.length; i++) {
            dayProfit[i] = prices[i] - prices[i - 1];
        }
        int profit = 0;
        for (int i = 1; i < prices.length; i++) {
            if (dayProfit[i] > 0) {
                profit = profit + dayProfit[i];
            }
        }
        return profit;
    }

    public int maxProfit3(int[] prices) {
        int profit = 0;
        for (int i = 0; i < prices.length; i++) {
            int profit1 = 0;
            int profit2 = 0;
            for (int j = 0; j <= i; j++) {
                for (int k = j + 1; k <= i; k++) {
                    profit1 = Math.max(profit1, prices[k] - prices[j]);
                }
            }
            for (int j = i + 1; j < prices.length; j++) {
                for (int k = j + 1; k < prices.length; k++) {
                    profit2 = Math.max(profit2, prices[k] - prices[j]);
                }
            }
            profit = Math.max(profit, profit1 + profit2);
        }
        return profit;
    }

    public int minimumTotal(ArrayList<ArrayList<Integer>> triangle) {
        if (triangle.size() == 0) return 0;
        ArrayList<ArrayList<Integer>> distance = new ArrayList<ArrayList<Integer>>(triangle);
        for (int i = distance.size() - 2; i >= 0; i--) {
            for (int j = 0; j <= i; j++) {
                distance.get(i).set(j, distance.get(i).get(j) +
                        Math.min(distance.get(i + 1).get(j), distance.get(i + 1).get(j + 1)));
            }

        }
        return distance.get(0).get(0);
    }

    public ArrayList<ArrayList<Integer>> generate(int numRows) {
        ArrayList<ArrayList<Integer>> list = new ArrayList<ArrayList<Integer>>();
        for (int i = 0; i < numRows; i++) {
            ArrayList<Integer> curList = new ArrayList<>();
            for (int j = 0; j <= i; j++) {
                if (j == 0 || j == i) curList.add(1);
                else curList.add(list.get(i - 1).get(j - 1) + list.get(i - 1).get(j));
            }
            list.add(curList);
        }
        return list;
    }

    public ArrayList<Integer> getRow(int rowIndex) {
        ArrayList<ArrayList<Integer>> list = new ArrayList<ArrayList<Integer>>();
        for (int i = 0; i <= rowIndex; i++) {
            ArrayList<Integer> curList = new ArrayList<>();
            for (int j = 0; j <= i; j++) {
                if (j == 0 || j == i) curList.add(1);
                else curList.add(list.get(i - 1).get(j - 1) + list.get(i - 1).get(j));
            }
            list.add(curList);
        }
        return list.get(rowIndex);
    }

    private class TreeLinkNode {
        int val;
        TreeLinkNode left, right, next;

        TreeLinkNode(int x) {
            val = x;
        }
    }

    public void connect(TreeLinkNode root) {
        if (root == null) return;
        TreeLinkNode node = null;
        Queue<TreeLinkNode> queue = new LinkedList<>();
        queue.offer(root);
        while (queue.isEmpty() == false) {
            int length = queue.size();//存储层次遍历时这层的长度
            for (int i = 0; i < length; i++) {
                node = queue.poll();
                if (i == length - 1) {//最右边的点，next指向null
                    node.next = null;
                } else {
                    node.next = queue.peek();
                }
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
        }
    }

    public int numDistinct(String S, String T) {
        if (S == null || T == null) return 0;
        if (T.length() > S.length()) return 0;
        int[][] dp = new int[S.length() + 1][T.length() + 1];
        dp[0][0] = 1;
        for (int i = 1; i <= S.length(); i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i <= S.length(); i++) {
            for (int j = 1; j <= T.length(); j++) {
                if (S.charAt(i - 1) != T.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j];
                } else if (S.charAt(i - 1) == T.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j] + dp[i][j];
                }
            }
        }
        return dp[S.length()][T.length()];
    }

    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) return false;
        if (root.left == null && root.right == null) {
            if (root.val == sum) return true;
            else return false;
        }
        if (root.left != null && hasPathSum(root.left, sum - root.val)) return true;
        if (root.right != null && hasPathSum(root.right, sum - root.val)) return true;
        return false;
    }

    public ArrayList<ArrayList<Integer>> pathSum(TreeNode root, int sum) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        ArrayList<Integer> list = new ArrayList<>();
        getPath(root, sum, list, result);
        return result;
    }

    private void getPath(TreeNode root, int sum, ArrayList<Integer> list, ArrayList<ArrayList<Integer>> result) {
        if (root == null) return;
        list.add(root.val);
        if (root.val == sum && root.left == null && root.right == null) {
            result.add(new ArrayList<Integer>(list));
        }
        if (root.left != null) {
            getPath(root.left, sum - root.val, list, result);
        }
        if (root.right != null) {
            getPath(root.right, sum - root.val, list, result);
        }
        list.remove(list.size() - 1);//不加这句就超时了
    }

    public boolean isBalanced(TreeNode root) {
        if (root == null) return true;
        int leftDepth = getTreeDepth(root.left);
        int rightDepth = getTreeDepth(root.right);
        if (Math.abs(leftDepth - rightDepth) <= 1) {
            if (isBalanced(root.left) && isBalanced(root.right)) {
                return true;
            }
        }
        return false;
    }

    private int getTreeDepth(TreeNode root) {
        if (root == null) return 0;
        if (root.left == null && root.right == null) return 1;
        return Math.max(getTreeDepth(root.left), getTreeDepth(root.right)) + 1;
    }

    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) return null;
        if (head.next == null) return new TreeNode(head.val);
        ListNode mid = head;
        ListNode end = head;
        ListNode preMid = null;
        while (end != null && end.next != null) {
            preMid = mid;
            mid = mid.next;
            end = end.next.next;//快慢指针找中点
        }
        TreeNode root = new TreeNode(mid.val);
        preMid.next = null;
        root.left = sortedListToBST(head);
        root.right = sortedListToBST(mid.next);
        return root;
    }

    public TreeNode sortedArrayToBST(int[] num) {
        if (num.length == 0) return null;
        return creatBST(num, 0, num.length - 1);
    }

    private TreeNode creatBST(int[] num, int start, int end) {
        if (start <= end) {
            int mid = (start + end) / 2 + (start + end) % 2;
            TreeNode root = new TreeNode(num[mid]);
            root.left = creatBST(num, start, mid - 1);
            root.right = creatBST(num, mid + 1, end);
            return root;
        }
        return null;
    }

    public ArrayList<ArrayList<Integer>> levelOrderBottom(TreeNode root) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (queue.isEmpty() == false) {
            ArrayList<Integer> list = new ArrayList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode curNode = queue.poll();
                list.add(curNode.val);
                if (curNode.left != null) queue.offer(curNode.left);
                if (curNode.right != null) queue.offer(curNode.right);
            }
            result.add(0, list);
        }
        return result;
    }

    public TreeNode buildTree(int[] inorder, int[] postorder) {
        if (inorder.length == 0 || postorder.length == 0) return null;
        return creatTree(inorder, 0, inorder.length - 1, postorder, 0, postorder.length - 1);
    }

    private TreeNode creatTree(int[] inorder, int inStart, int inEnd,
                               int[] postorder, int postStart, int postEnd) {
        if (inStart > inEnd || postStart > postEnd) return null;
        TreeNode root = new TreeNode(postorder[postEnd]);
        for (int i = 0; i < postorder.length; i++) {
            if (inorder[i] == postorder[postEnd]) {//中序中找到根节点，处理左右孩子
                root.left = creatTree(inorder, inStart, i - 1, postorder, postStart, postStart - inStart + i - 1);
                root.right = creatTree(inorder, i + 1, inEnd, postorder, postStart - inStart + i, postEnd - 1);
            }
        }
        return root;
    }

    private TreeNode creatTree2(int[] preorder, int preStart, int preEnd,
                                int[] inorder, int inStart, int inEnd) {
        if (preStart > preEnd || inStart > inEnd) return null;
        TreeNode root = new TreeNode(preorder[preStart]);
        for (int i = 0; i < inorder.length; i++) {
            if (inorder[i] == preorder[preStart]) {//中序中找到根节点，处理左右孩子
                root.left = creatTree2(preorder, preStart + 1, preStart - inStart + i, inorder, inStart, i - 1);
                root.right = creatTree2(preorder, preStart - inStart + i + 1, preEnd, inorder, i + 1, inEnd);
            }
        }
        return root;
    }

    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        if (root.left == null && root.right == null) return 1;
        if (root.left == null) return maxDepth(root.right) + 1;
        if (root.right == null) return maxDepth(root.left) + 1;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    public ArrayList<ArrayList<Integer>> zigzagLevelOrder(TreeNode root) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (queue.isEmpty() == false) {
            ArrayList<Integer> list = new ArrayList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode curNode = queue.poll();
                list.add(curNode.val);
                if (curNode.left != null) queue.offer(curNode.left);
                if (curNode.right != null) queue.offer(curNode.right);
            }
            result.add(list);
        }
        //处理锯齿形的输出
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        int count = 0;
        while (count < result.size()) {
            //一行正输出，一行反输出
            ArrayList<Integer> list = new ArrayList<>();
            if (count % 2 == 0) {
                for (int i = 0; i < result.get(count).size(); i++) {
                    list.add(result.get(count).get(i));
                }
            } else if (count % 2 == 1) {
                for (int i = 0; i < result.get(count).size(); i++) {
                    list.add(0, result.get(count).get(i));
                }
            }
            count++;
            res.add(list);
        }
        return res;
    }

    public ArrayList<ArrayList<Integer>> levelOrder(TreeNode root) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (queue.isEmpty() == false) {
            ArrayList<Integer> list = new ArrayList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode curNode = queue.poll();
                list.add(curNode.val);
                if (curNode.left != null) queue.offer(curNode.left);
                if (curNode.right != null) queue.offer(curNode.right);
            }
            result.add(list);
        }
        return result;
    }

    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return isSymmetricHelper(root.left, root.right);
    }

    private boolean isSymmetricHelper(TreeNode leftChild, TreeNode rightChild) {
        if (leftChild == null && rightChild == null) return true;
        if (leftChild == null && rightChild != null) return false;
        if (leftChild != null && rightChild == null) return false;
        if (leftChild.val != rightChild.val) return false;
        return isSymmetricHelper(leftChild.left, rightChild.right) && isSymmetricHelper(leftChild.right, rightChild.left);
    }

    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        if (p == null && q != null) return false;
        if (p != null && q == null) return false;
        if (p.val != q.val) return false;
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

    public void recoverTree(TreeNode root) {
        ArrayList<TreeNode> list = new ArrayList<>();
        getBSTNode(root, list);
        TreeNode mistake1 = null;
        TreeNode mistake2 = null;
        for (int i = 0; i < list.size() - 1; i++) {
            if (list.get(i).val > list.get(i + 1).val) {
                mistake1 = list.get(i);
                break;
            }
        }
        for (int i = list.size() - 1; i > 0; i--) {
            if (list.get(i).val < list.get(i - 1).val) {
                mistake2 = list.get(i);
                break;
            }
        }
        int temp = mistake1.val;
        mistake1.val = mistake2.val;
        mistake2.val = temp;
    }

    private void getBSTNode(TreeNode root, ArrayList<TreeNode> list) {
        if (root != null) {
            getBSTNode(root.left, list);
            list.add(root);
            getBSTNode(root.right, list);
        }
    }

    public boolean isValidBST(TreeNode root) {
        ArrayList<TreeNode> list = new ArrayList<>();
        getBSTNode(root, list);
        for (int i = 1; i < list.size(); i++) {
            if (list.get(i - 1).val >= list.get(i).val) return false;
        }
        return true;
    }

    public boolean isInterleave(String s1, String s2, String s3) {
        if (s1.length() + s2.length() != s3.length()) return false;
        boolean[][] dp = new boolean[s1.length() + 1][s2.length() + 1];
        dp[0][0] = true;
        for (int i = 1; i <= s1.length(); i++) {
            dp[i][0] = dp[i - 1][0] && s1.charAt(i - 1) == s3.charAt(i - 1);
        }
        for (int i = 1; i <= s2.length(); i++) {
            dp[0][i] = dp[0][i - 1] && s2.charAt(i - 1) == s3.charAt(i - 1);
        }
        for (int i = 1; i <= s1.length(); i++) {
            for (int j = 1; j <= s2.length(); j++) {
                if (dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1)) dp[i][j] = true;
                else if (dp[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1)) dp[i][j] = true;
                else dp[i][j] = false;
            }
        }
        return dp[s1.length()][s2.length()];
    }

    public int numTrees(int n) {
        if (n <= 1) return 1;
        int count = 0;
        for (int i = 1; i <= n; i++) {
            count = count + numTrees(i - 1) * numTrees(n - i);
        }
        return count;
    }

    public ArrayList<TreeNode> generateTrees(int n) {
        return creatBST2(1, n);
    }

    private ArrayList<TreeNode> creatBST2(int low, int high) {
        ArrayList<TreeNode> result = new ArrayList<>();
        if (low > high) {
            result.add(null);
            return result;
        }
        for (int i = low; i <= high; i++) {
            ArrayList<TreeNode> left = creatBST2(low, i - 1);
            ArrayList<TreeNode> right = creatBST2(i + 1, high);
            for (int j = 0; j < left.size(); j++) {
                for (int k = 0; k < right.size(); k++) {//和第一题一样，左子树*右子树
                    TreeNode node = new TreeNode(i);
                    node.left = left.get(j);
                    node.right = right.get(k);
                    result.add(node);
                }
            }
        }
        return result;
    }

    public ArrayList<Integer> inorderTraversal(TreeNode root) {
        ArrayList<Integer> inOrder = new ArrayList<>();
        if (root == null) return inOrder;
        getInorder(root, inOrder);
        return inOrder;
    }

    private void getInorder(TreeNode root, ArrayList<Integer> inOrder) {
        if (root == null) return;
        else {
            getInorder(root.left, inOrder);
            inOrder.add(root.val);
            getInorder(root.right, inOrder);
        }
    }

    public ArrayList<String> restoreIpAddresses(String s) {
        ArrayList<String> result = new ArrayList<>();
        if (s.length() > 12 || s.length() == 0) return result;
        getIpAddresses(s, 0, "", result);
        return result;
    }

    private void getIpAddresses(String input, int position, String ipAddress, ArrayList<String> result) {
        if (input.length() == 0) return;
        if (position == 3) {//只能存在四节ip
            int num = Integer.parseInt(input);
            if (input.charAt(0) == '0') {
                if ((input.length() == 1 && num == 0) == false) return;
            }
            if (num <= 255) {
                ipAddress = ipAddress + input;
                result.add(ipAddress);
                return;
            }
        } else {
            if (input.length() >= 1) {
                getIpAddresses(input.substring(1), position + 1,
                        ipAddress + input.substring(0, 1) + ".", result);
            }
            if (input.length() >= 2 && input.charAt(0) != '0') {
                getIpAddresses(input.substring(2), position + 1,
                        ipAddress + input.substring(0, 2) + ".", result);
            }
            if (input.length() >= 3 && input.charAt(0) != '0') {
                int num = Integer.parseInt(input.substring(0, 3));
                if (num <= 255) {
                    getIpAddresses(input.substring(3), position + 1,
                            ipAddress + input.substring(0, 3) + ".", result);
                }
            }
        }
    }

    public ListNode reverseBetween(ListNode head, int m, int n) {
        if (head == null) return null;
        ListNode preHead = new ListNode(0);
        preHead.next = head;//保留最初的head
        ListNode preStart = preHead;
        ListNode start = head;
        for (int i = 1; i < m; i++) {
            preStart = start;
            start = start.next;
        }//find m position
        for (int i = 0; i < n - m; i++) {
            ListNode temp = start.next;
            start.next = temp.next;
            temp.next = preStart.next;
            preStart.next = temp;
        }
        return preHead.next;
    }

    public ArrayList<ArrayList<Integer>> subsets(int[] S) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (S.length == 0) return result;
        ArrayList<Integer> list = new ArrayList<>();
        Arrays.sort(S);
        dfsSubsets(S, 0, list, result);
        return result;
    }

    private void dfsSubsets(int[] num, int start, ArrayList<Integer> list, ArrayList<ArrayList<Integer>> result) {
        result.add(new ArrayList<Integer>(list));
        for (int i = start; i < num.length; i++) {
            list.add(num[i]);
            dfsSubsets(num, i + 1, list, result);
            list.remove(list.size() - 1);
        }
    }

    public ArrayList<ArrayList<Integer>> subsetsWithDup(int[] num) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        if (num.length == 0) return result;
        ArrayList<Integer> list = new ArrayList<>();
        Arrays.sort(num);
        dfsSubsets2(num, 0, list, result);
        return result;
    }

    private void dfsSubsets2(int[] num, int start, ArrayList<Integer> list, ArrayList<ArrayList<Integer>> result) {
        result.add(new ArrayList<Integer>(list));
        for (int i = start; i < num.length; i++) {
            list.add(num[i]);
            dfsSubsets2(num, i + 1, list, result);
            list.remove(list.size() - 1);
            while (i < num.length - 1 && num[i] == num[i + 1])//跳过重复元素
                i++;
        }
    }

    public int numDecodings(String s) {
        if (s.length() == 0 || s.charAt(0) == '0') return 0;
        if (s.length() == 1 && s.charAt(0) != '0') return 1;
        int[] dp = new int[s.length() + 1];
        dp[0] = dp[1] = 1;
        for (int i = 2; i <= s.length(); i++) {
            int num = Integer.parseInt(s.substring(i - 2, i));
            if (10 <= num && num <= 26) dp[i] = dp[i] + dp[i - 2];
            if (s.charAt(i - 1) != '0') dp[i] = dp[i] + dp[i - 1];
        }
        return dp[s.length()];
    }

    public ArrayList<Integer> grayCode(int n) {
        ArrayList<Integer> result = new ArrayList<>();
        int num = (int) Math.pow(2, n);
        for (int i = 0; i < num; i++) {
            result.add(i >> 1 ^ i);
        }
        return result;
    }

    public void merge(int A[], int m, int B[], int n) {
        for (int i = 0; i < n; i++) {
            A[i + m] = B[i];
        }
        Arrays.sort(A);
    }

    public boolean isScramble(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        int len = s1.length();
        boolean[][][] dp = new boolean[len + 1][len + 1][len + 1];
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < len; j++) {
                if (s1.charAt(i) == s2.charAt(j))
                    dp[1][i][j] = true;
                else dp[1][i][j] = false;
            }
        }
        for (int k = 2; k <= len; ++k) {
            for (int i = 0; i <= len - k; ++i) {
                for (int j = 0; j <= len - k; ++j) {
                    //div表示长度为k的子串中，将子串一分为二的分割点
                    for (int div = 1; div < k && !dp[k][i][j]; ++div) {
                        // dp[k][i][j] = true的条件是子串分割后的两段对应相等，或者交叉对应相等
                        if ((dp[div][i][j] && dp[k - div][i + div][j + div])
                                || (dp[div][i][j + k - div] && dp[k - div][i + div][j])) {
                            dp[k][i][j] = true;
                        }
                    }
                }
            }
        }
        return dp[len][0][0];
    }

    public ListNode partition(ListNode head, int x) {
        if (head == null) return null;
        ListNode preHead1 = new ListNode(Integer.MIN_VALUE);
        ListNode preHead2 = new ListNode(Integer.MIN_VALUE);
        ListNode curNode1 = preHead1;
        ListNode curNode2 = preHead2;
        while (head != null) {
            if (head.val < x) {
                curNode1.next = head;
                curNode1 = curNode1.next;
            } else {
                curNode2.next = head;
                curNode2 = curNode2.next;
            }
            head = head.next;
        }
        curNode2.next = null;
        curNode1.next = preHead2.next;
        return preHead1.next;
    }

    public int largestRectangleArea(int[] height) {
        if (height.length == 0) return 0;
        Stack<Integer> stack = new Stack<>();
        int result = 0;
        for (int i = 0; i < height.length; i++) {
            if (stack.isEmpty() || stack.peek() <= height[i]) stack.push(height[i]);
            else {
                int count = 0;
                while (!stack.isEmpty() && stack.peek() > height[i]) {
                    count++;
                    result = Math.max(result, stack.peek() * count);
                    stack.pop();
                }
                while (count > 0) {
                    count--;
                    stack.push(height[i]);
                }
                stack.push(height[i]);
            }
        }
        int depth = 1;
        while (!stack.isEmpty()) {
            result = Math.max(result, stack.peek() * depth);
            stack.pop();
            depth++;
        }
        return result;
    }

    public int maximalRectangle(char[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0) return 0;
        int row = matrix.length;
        int col = matrix[0].length;
        int result = 0;
        for (int i = 0; i < row; i++) {
            int[] num = new int[col];
            Arrays.fill(num, 0);
            for (int j = 0; j < col; j++) {
                int k = i;
                while (k >= 0 && matrix[k][j] == '1') {
                    num[j]++;
                    k--;
                }

            }
            result = Math.max(result, largestRectangleArea(num));
        }
        return result;
    }

    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) return null;
        ListNode curNode = head;
        while (curNode != null) {
            while (curNode.next != null && curNode.val == curNode.next.val) {
                curNode.next = curNode.next.next;
            }
            curNode = curNode.next;
        }
        return head;
    }

    public ListNode deleteDuplicates2(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode preNode = new ListNode(0);
        ListNode curNode = head;
        preNode.next = head;
        ListNode node = preNode;
        while (curNode != null && curNode.next != null) {
            if (curNode.val != curNode.next.val) {
                node = curNode;
            } else {
                while (curNode.next != null && curNode.val == curNode.next.val) {
                    curNode = curNode.next;
                    node.next = curNode.next;
                }
            }
            curNode = curNode.next;
        }
        return preNode.next;
    }

    public int search(int[] A, int target) {
        int position = 0;
        for (int i = 1; i < A.length; i++) {
            if (A[i] < A[i - 1]) {
                position = i - 1;
            }
        }
        int low = 0;
        int high = position;
        while (low <= high) {
            int mid = (low + high) / 2;
            if (A[mid] == target) return mid;
            else if (A[mid] > target) high = mid - 1;
            else if (A[mid] < target) low = mid + 1;
        }
        low = position + 1;
        high = A.length - 1;
        while (low <= high) {
            int mid = (low + high) / 2;
            if (A[mid] == target) return mid;
            else if (A[mid] > target) high = mid - 1;
            else if (A[mid] < target) low = mid + 1;
        }
        return -1;
    }

}
