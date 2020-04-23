import java.rmi.Remote;
import java.rmi.RemoteException;
public interface Hello extends Remote{
    // 메소드 하나를 선언만 함
    public String sayHello(String name) throws RemoteException; // sayHello: 내가 서비스 해야 할 메소드. 원격에서 호출할 거라고 기대되는 메소드

}
