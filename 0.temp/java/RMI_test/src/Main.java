import java.rmi.*;
import java.rmi.ServerError.*;

public class Main {

    public static void main(String[] args) throws Exception{
        HelloImpl h = new HelloImpl(); //throws Exception이 없으면 오류가 난다.
        Naming.bind("//localhost/hello", h);
        System.out.println("Hello. The server is ready.");
        //System.out.println("Hello World!");
    }
}
