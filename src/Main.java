import java.util.*;

public class Main {

    public static void main(String[] args) {
        String[] tokens = new String[]{"2", "1", "+", "3", "*"};
        Solution s = new Solution();
        int result=s.evalRPN(tokens);
        System.out.println(result);
    }


}
