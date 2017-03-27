import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int a = in.nextInt();
        int b = in.nextInt();
        int dif=a^b;
        int count = 0;
        while(dif != 0){
            count++;
            dif &= (dif-1);
            System.out.println(dif);
            System.out.println(dif-1);
        }
        System.out.println(count);
    }
}
