import java.util.*;

public class Main {

    public static void main(String[] args) {
//        Solution s = new Solution();
        Scanner in = new Scanner(System.in);

        while (in.hasNext()) {
            String str = in.nextLine();
            char[] pw = str.toCharArray();
            int a1, a2, a3, a4;
            a1 = a2 = a3 = a4 = 0;
            if (str.length() < 8) {
                System.out.println("NG");
                continue;
            }
            if (check(str) == false) {
                System.out.println("NG");
                continue;
            }
            for (int i = 0; i < pw.length; i++) {
                if (pw[i] >= '0' && pw[i] <= '9') {
                    a1 = 1;
                    continue;
                }
                if (pw[i] >= 'a' && pw[i] <= 'z') {
                    a2 = 1;
                    continue;
                }
                if (pw[i] >= 'A' && pw[i] <= 'Z') {
                    a3 = 1;
                    continue;
                } else {
                    a4 = 1;
                    continue;
                }
            }
            int total = a1 + a2 + a3 + a4;
            if (total >= 3) {
                System.out.println("OK");
            } else {
                System.out.println("NG");
            }


        }
    }

    public static boolean check(String str) {
        for (int i = 0; i < str.length() - 2; i++) {
            String str1 = str.substring(i, i + 3);
            if (str.substring(i + 1).contains(str1)) {
                return false;
            }
        }
        return true;
    }
}
