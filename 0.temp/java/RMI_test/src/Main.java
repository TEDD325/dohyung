import java.rmi.*;
import java.rmi.ServerError.*;

public class Main {

    public static void main(String[] args) throws Exception{
        HelloImpl_ h = new HelloImpl_(); //throws Exception이 없으면 오류가 난다.
        Naming.bind("//localhost/hello", h); //브로커 역할. localhost; 브로커의 위치
        System.out.println("Hello. The server is ready.");
        //System.out.println("Hello World!");
    }
}
