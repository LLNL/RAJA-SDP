// g++ --std=c++11 main-tye_erasure.cpp
#include <iostream>
#include <memory>

namespace exp{
  class GPU
  {
    public:
      void print_name() const { std::cout << "I am GPU " << value << std::endl; }
      void do_GPU() { std::cout << "Do GPU specific member function" << std::endl; }
      void set_value(int i) { value = i; }
    private:
      int value = 1;
  };

  class Host
  {
    public:
      void print_name() const { std::cout << "I am Host " << value << std::endl; }
      void do_Host() { std::cout << "Do Host specific member funtion." << std::endl; }
      void set_value(int i) { value = i; }
    private:
      int value = 1;
  };


  class Context
  {
    public:
      template<typename T>
      Context(T&& value){ *this = value; }
      
      template<typename T>
      Context& operator=(T&& value) { m_value.reset(new ContextModel<T>(value)); }

      void print_name() const { m_value->print_name(); }
      void set_value(int i) const { m_value->set_value(i); }

      template<typename T>
      T& get_device() { static_cast<T*>(m_value->get_device()); }

      template<typename T>
      void get_device(T *dst) { dst = static_cast<T*>(m_value->get_device()); }

    private:
      class ContextConcept {
	public:
	  virtual ~ContextConcept(){}
	  virtual void print_name() const = 0;
	  virtual void set_value(int i) = 0;
	  virtual void* get_device() = 0;
      };

      template<typename T>
      class ContextModel : public ContextConcept {
	public:
	  ContextModel(T const& modelVal) : m_modelVal(modelVal) {}
	  void print_name() const override { m_modelVal.print_name(); }
	  void set_value(int i) override { m_modelVal.set_value(i); }
	  void *get_device() override { return &m_modelVal; }
	private:
	  T m_modelVal;
      };

      std::unique_ptr<ContextConcept> m_value;
  };
}


int main(int argc, char*argv[])
{
  using namespace exp;
  std::cout << "- TEST : Creating GPU context" << std::endl;
  Context my_dev{GPU()};
  my_dev.print_name();

  std::cout << "- TEST : Setting my_dev to Host object" << std::endl; 
  my_dev = Host();
  my_dev.print_name();

  std::cout << "- TEST : Copy Constructor test" << std::endl; 
  Context my_dev2{GPU()};
  my_dev.set_value(2);
  my_dev.print_name();
  my_dev2 = my_dev;
  my_dev2.print_name();

  std::cout << "- TEST : Getting Typed object from context (copy constructed)" << std::endl;
  Host h = my_dev.get_device<Host>(); 
  h.do_Host();
  h.print_name();
  h.set_value(3);

  Host h2;
  my_dev.get_device(&h2);
//  h2 = my_dev.get_device<Host>();
  h2.do_Host();

  h.print_name();
  h2.print_name(); // should print 2, instead prints 1

  my_dev.print_name();
  return 0;
}
