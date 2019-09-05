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

      Context(){}

      template<typename T>
      Context(T&& value){ m_value.reset(new ContextModel<T>(value));}

      void print_name() const { m_value->print_name(); }
      void set_value(int i) const { m_value->set_value(i); }

      template<typename T>
      T* get_device() { 
	auto result = dynamic_cast<ContextModel<T>*>(m_value.get()); 
	if (result ==nullptr)
	{
	  std::cout << "NULLPTR" << std::endl;
	  std::exit(1);
	}
	return result->get_device();
      }

    private:
      class ContextConcept {
	public:
	  virtual ~ContextConcept(){}
	  virtual void print_name() const = 0;
	  virtual void set_value(int i) = 0;
      };

      template<typename T>
      class ContextModel : public ContextConcept {
	public:
	  ContextModel(T const& modelVal) : m_modelVal(modelVal) {}
	  void print_name() const override { m_modelVal.print_name(); }
	  void set_value(int i) override { m_modelVal.set_value(i); }
	  T *get_device() { return &m_modelVal; }
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
  Context dev{GPU()};
  dev.print_name();

  std::cout << "- TEST : Setting dev to Host object" << std::endl; 
  dev = Host();
  dev.print_name();

  std::cout << "- TEST : Getting Typed object from context (copy constructed)" << std::endl;
  auto h = dev.get_device<Host>(); 
  h->do_Host();

  Context gpu_dev;
  gpu_dev = GPU();
  gpu_dev.print_name();
  auto g = gpu_dev.get_device<GPU>();
  g->do_GPU();

  g->set_value(22); 
  g->print_name();
  gpu_dev.print_name();

  return 0;
}
