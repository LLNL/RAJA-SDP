#ifndef __CAMP_DEVICES_HPP
#define __CAMP_DEVICES_HPP

#include <cstring>
#include <memory>

#include <cuda_runtime.h>
namespace camp
{
namespace devices
{

  enum class Platform {
    undefined = 0,
    host = 1,
    omp = 2,
    tbb = 4,
    cuda = 8,
    // omp_target = 16, not sure this is a meaningful difference
    hip = 32
  };


  class Event
  {
  public:
    Event(){ cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming); }
    void capture(cudaStream_t stream){ cudaEventRecord(m_event, stream); }
    void wait(){ while(cudaEventQuery(m_event) != 0){} } 
  private:
    cudaEvent_t m_event;
  };


  class Cuda 
  {
    static cudaStream_t get_a_stream(int num)
    {
      // TODO consider pool size
      static cudaStream_t streams[16] = {};
      static int previous = 0;
      // TODO deal with parallel init
      if (streams[0] == nullptr) {
        for (auto &s : streams) {
          cudaStreamCreate(&s);
        }
      }

      if (num < 0) {
        previous = (previous + 1) % 16;
        return streams[previous];
      }

      return streams[num % 16];
    }

  public:
    Cuda(int device = 0, int group = -1) : stream(get_a_stream(group)) {}

    // Methods
    Platform get_platform() { return Platform::cuda; }
    static Cuda &get_default()
    {
      static Cuda h;
      return h;
    }
    void wait() { cudaStreamSynchronize(stream); }
    void wait_on(Event e) { e.wait(); }
    template <typename T>
    T *allocate(size_t size)
    {
      T *ret = nullptr;
      cudaMallocManaged(&ret, sizeof(T) * size);
      return ret;
    }
    void *calloc(size_t size)
    {
      void *p = allocate<char>(size);
      this->memset(p, 0, size);
      return p;
    }
    void free(void *p) { cudaFree(p); }
    void memcpy(void *dst, const void *src, size_t size)
    {
      cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream);
    }
    void memset(void *p, int val, size_t size)
    {
      cudaMemsetAsync(p, val, size, stream);
    }

    cudaStream_t get_stream() { return stream; }

  private:
    cudaStream_t stream;
  };

  class Host
  {
  public:
    Host(int device = 0, int group = -1) {}

    // Methods
    Platform get_platform() { return Platform::host; }
    Host &get_default()
    {
      static Host h;
      return h;
    }
    void wait()
    {
      // nothing to wait for, sequential/simd host is always synchronous
    }
    // Memory
    template <typename T>
    T *allocate(size_t n)
    {
      return (T*)malloc(sizeof(T) * n);
    }
    void *calloc(size_t size)
    {
      void *p = allocate<char>(size);
      this->memset(p, 0, size);
      return p;
    }
    void free(void *p) { free(p); }
    void memcpy(void *dst, const void *src, size_t size)
    {
      memcpy(dst, src, size);
    }
    void memset(void *p, int val, size_t size) { std::memset(p, val, size); }
  };


  class Context
  {
    public:

      Context(){}

      template<typename T>
      Context(T&& value){ m_value.reset(new ContextModel<T>(value));}

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
      Platform get_platform() { return m_value->get_platform(); }
      template <typename T>
      T *allocate(size_t size)
      {
	return (T *)m_value->calloc(size * sizeof(T));
      }
      void *calloc(size_t size) { return m_value->calloc(size); }
      void free(void *p) { m_value->free(p); }
      void memcpy(void *dst, const void *src, size_t size)
      {
	m_value->memcpy(dst, src, size);
      }
      void memset(void *p, int val, size_t size) { m_value->memset(p, val, size); }
      void wait_on(Event e) { m_value->wait_on(e); }


    private:
      class ContextConcept {
	public:
	  virtual ~ContextConcept(){}
	  virtual Platform get_platform() = 0;
	  virtual void *calloc(size_t size) = 0;
	  virtual void free(void *p)=0;
	  virtual void memcpy(void *dst, const void *src, size_t size)=0;
	  virtual void memset(void *p, int val, size_t size)=0;
	  virtual void wait_on(Event e)=0;
      };

      template<typename T>
      class ContextModel : public ContextConcept {
	public:
	  ContextModel(T const& modelVal) : m_modelVal(modelVal) {}
          Platform get_platform() override { return m_modelVal.get_platform(); }
	  void *calloc(size_t size) override { return m_modelVal.calloc(size); }
	  void free(void *p) override { m_modelVal.free(p); }
	  void memcpy(void *dst, const void *src, size_t size) override
	  {
	    m_modelVal.memcpy(dst, src, size);
	  }
	  void memset(void *p, int val, size_t size) override
	  {
	    m_modelVal.memset(p, val, size);
	  }
	  void wait_on(Event e) { m_modelVal.wait_on(e); }
	  T *get_device() { return &m_modelVal; }
	private:
	  T m_modelVal;
      };

      std::unique_ptr<ContextConcept> m_value;
  };

  
/*
  class Omp : public Host
  {
    // TODO: see if using fake addresses is an issue
    char *dep = nullptr;

  public:
    Omp(int device = 0, int group = -1) : dep((char *)group) {}

    // Methods
    Platform get_platform() { return Platform::omp; }

    Omp &get_default()
    {
      static Omp h;
      return h;
    }

    void wait()
    {
// TODO: see if taskwait depend has wide enough support
#pragma omp task if (0) depend(dep[0])
      {
      }
      // #pragma omp taskwait depend(dep[0])
    }

    char *get_dep() { return dep; }

    // Memory: inherited from Host
    void memset(void *p, int val, size_t size)
    {
      if (omp_get_level() != 0) {
        ::std::memset(p, val, size);
      } else {
        char *c = (char *)p;
#pragma omp parallel for simd
        for (size_t i = 0; i < size; ++i) {
          c[i] = val;
        }
      }
    }
  };
*/

}  // namespace devices
}  // namespace camp
#endif /* __CAMP_DEVICES_HPP */
