#include <iostream>
class Shape {
public:
  virtual double getArea() = 0; // pure virtual function
};

// Derived class
class Circle : public Shape {
public:
  Circle(double radius) : radius(radius) {}
  virtual double getArea() override {
    return 3.14 * radius * radius;
  }
private:
  double radius;
};

// Class accepting Shape object
class ShapeCalculations {
public:
  ShapeCalculations(Shape* shape) : shape(shape) {}
  double getArea() {
    return shape->getArea();
  }
private:
  Shape* shape;
};

// Usage example
int main() {
  Circle circle(5);
  ShapeCalculations calc(&circle);
  std::cout << "Circle area: " << calc.getArea() << std::endl; // Output: Circle area: 78.5
    return 0;
}
