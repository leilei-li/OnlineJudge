import java.util.*;

public class Main {

    public static void main(String[] args) {
        String[] tokens = new String[]{"2", "1", "+", "3", "*"};
        LeetCode1 leetCode1 = new LeetCode1();
        int[] price = new int[]{1, 2, 4};
        //leetCode1.minCut("ab");
        System.out.println(leetCode1.generate(3));
    }


}
