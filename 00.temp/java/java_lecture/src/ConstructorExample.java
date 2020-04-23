public class ConstructorExample{
    int age;
    String name;

    ConstructorExample(){
        this.name = "Chaitanya";
        this.age = 30;
    }

    ConstructorExample(String n, int a){
        this.name = n;
        this.age = a;
    }

    public static void main(String args[]){
        ConstructorExample obj1 = new ConstructorExample();
        ConstructorExample obj2 = new ConstructorExample("steve", 56);

        System.out.println(obj1.name + " " + obj1.age);
        System.out.println(obj2.name + " " + obj2.age);
    }
}
