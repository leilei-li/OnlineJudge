import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        String str=in.nextLine();
        int n=in.nextInt();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if(str.charAt(i)==str.charAt(j)){
                    System.out.println(str.charAt(i));
                    break;
                }
            }
        }
    }
}
