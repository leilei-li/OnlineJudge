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

}
