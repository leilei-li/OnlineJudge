import sun.nio.cs.ext.GBK;

import java.io.*;
import java.util.*;

public class Main {

    public static void main(String[] args) {
//        Solution s=new Solution();
//        Scanner in = new Scanner(System.in);
//        while (in.hasNext()) {
//            String str=in.nextLine();
//            str=str.substring(2);
//            System.out.println(Integer.parseInt(str,16));
//        }
        File file = new File("/Users/lileilei/Downloads/超品相师.txt");
        try {
            InputStreamReader read = new InputStreamReader(new FileInputStream(file), "GBK");
            BufferedReader bufferedReader = new BufferedReader(read);
            String str = bufferedReader.readLine();
            File outFile = new File("/Users/lileilei/Downloads/1.txt");
            FileOutputStream out = new FileOutputStream(outFile, true); //如果追加方式用true
            while (str != null) {
                str = bufferedReader.readLine();
                String nameGBK = str;
                byte[] temp = nameGBK.getBytes("GBK");
                byte[] newtemp = new String(temp, "GBK").getBytes("utf8");
                String strUtf8 = new String(newtemp, "utf8");
                System.out.println(strUtf8);
                out.write(newtemp);
            }
        } catch (Exception e) {
            System.out.println("read fail!");
        }

    }
}
