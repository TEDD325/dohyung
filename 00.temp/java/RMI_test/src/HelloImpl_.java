import java.rmi.*;
import java.rmi.server.*;

public class HelloImpl_ extends UnicastRemoteObject implements Hello{
    // 구체사항들은 UnicastRemoteObject에 구현되어 있다.
    // 생성자를 만들어 주어야 한다.
    protected HelloImpl_() throws RemoteException{

    }

    public String sayHello(String name) throws RemoteException{
        return "Hello World! from " + name;
    }
}
