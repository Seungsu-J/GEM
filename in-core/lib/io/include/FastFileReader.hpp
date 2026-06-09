#ifndef IO_FAST_FILE_READER_H
#define IO_FAST_FILE_READER_H

#include <cstdio>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector> // Used for read_string

// Fast I/O optimization constants
constexpr size_t BUFFER_SIZE = 256 * 1024 * 1024; // 256MB buffer for file I/O, for large file (data graph)

class FastFileReader
{
private:
  FILE *file_;
  std::unique_ptr<char[]> buffer_;
  size_t buffer_size_;
  size_t pos_;
  size_t end_;

  void fill_buffer()
  {
    end_ = fread(buffer_.get(), 1, buffer_size_, file_);
    pos_ = 0;
  }

public:
  explicit FastFileReader(const char *filename)
      : file_(fopen(filename, "rb")), buffer_(std::make_unique<char[]>(BUFFER_SIZE)),
        buffer_size_(BUFFER_SIZE), pos_(0), end_(0)
  {
    if (!file_)
    {
      throw std::runtime_error(std::string("Unable to open file: ") + filename);
    }
    fill_buffer();
  }

  ~FastFileReader()
  {
    if (file_)
      fclose(file_);
  }

  // --- FIX: Explicitly define move semantics for robust resource management ---
  FastFileReader(FastFileReader &&other) noexcept
      : file_(other.file_), buffer_(std::move(other.buffer_)),
        buffer_size_(other.buffer_size_), pos_(other.pos_), end_(other.end_)
  {
    other.file_ = nullptr; // Prevent double-free
  }

  FastFileReader &operator=(FastFileReader &&other) noexcept
  {
    if (this != &other)
    {
      if (file_)
        fclose(file_);
      file_ = other.file_;
      buffer_ = std::move(other.buffer_);
      buffer_size_ = other.buffer_size_;
      pos_ = other.pos_;
      end_ = other.end_;
      other.file_ = nullptr;
    }
    return *this;
  }

  // Prevent copying
  FastFileReader(const FastFileReader &) = delete;
  FastFileReader &operator=(const FastFileReader &) = delete;

  char next_char()
  {
    if (pos_ >= end_)
    {
      fill_buffer();
      if (end_ == 0)
        return EOF;
    }
    return buffer_[pos_++];
  }

  char peek_char()
  {
    if (pos_ >= end_)
    {
      fill_buffer();
      if (end_ == 0)
        return EOF;
    }
    return buffer_[pos_];
  }

  // --- FIX: Renamed to read_integer for clarity and improved logic ---
  template <typename T>
  T read_integer()
  {
    T result = 0;
    char c = peek_char();

    // Skip whitespace
    while (c == ' ' || c == '\t' || c == '\n' || c == '\r')
    {
      next_char(); // Consume whitespace
      c = peek_char();
    }

    if (c == EOF)
      throw std::runtime_error("Unexpected EOF while reading integer.");

    bool negative = false;
    if (c == '-')
    {
      negative = true;
      next_char(); // Consume '-'
    }
    else if (c == '+')
    {
      next_char(); // Consume '+'
    }

    // --- FIX: Use peek_char() to avoid consuming the delimiter ---
    while (peek_char() >= '0' && peek_char() <= '9')
    {
      result = result * 10 + (next_char() - '0');
    }

    return negative ? -result : result;
  }

  // --- ADD: Function to read floating-point numbers ---
  double read_double()
  {
    double result = 0.0;
    char c = peek_char();

    // Skip whitespace
    while (c == ' ' || c == '\t' || c == '\n' || c == '\r')
    {
      next_char();
      c = peek_char();
    }
    if (c == EOF)
      throw std::runtime_error("Unexpected EOF while reading double.");

    bool negative = false;
    if (c == '-')
    {
      negative = true;
      next_char();
    }
    else if (c == '+')
    {
      next_char();
    }

    while (peek_char() >= '0' && peek_char() <= '9')
    {
      result = result * 10.0 + (next_char() - '0');
    }

    if (peek_char() == '.')
    {
      next_char(); // Consume '.'
      double fraction = 0.1;
      while (peek_char() >= '0' && peek_char() <= '9')
      {
        result += (next_char() - '0') * fraction;
        fraction *= 0.1;
      }
    }
    // Note: Scientific notation (e.g., 1.23e4) is not handled here for simplicity.
    return negative ? -result : result;
  }

  // --- ADD: Function to read a whitespace-separated string ---
  std::string read_string()
  {
    std::string str;
    char c = peek_char();

    // Skip leading whitespace
    while (c == ' ' || c == '\t' || c == '\n' || c == '\r')
    {
      next_char();
      c = peek_char();
    }

    if (c == EOF)
      throw std::runtime_error("Unexpected EOF while reading string.");

    // Read until next whitespace
    while (c != EOF && c != ' ' && c != '\t' && c != '\n' && c != '\r')
    {
      str += next_char();
      c = peek_char();
    }
    return str;
  }

  char read_char()
  {
    if (pos_ >= end_)
    {
      fill_buffer();
      if (end_ == 0)
        return EOF;
    }
    return buffer_[pos_++];
  }
};

#endif // IO_FAST_FILE_READER_H