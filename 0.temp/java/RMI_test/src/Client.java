import java.rmi.*;

public class Client {
    public static void main(String[] args) throws Exception{
        Hello h = (Hello)Naming.lookup("//localhost/hello");
        String messages = h.sayHello("dohk");
        System.out.println(messages);
    }
}
