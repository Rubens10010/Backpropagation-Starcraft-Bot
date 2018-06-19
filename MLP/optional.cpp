#include <iostream>
#include <optional>

class temp
{
	public:
	temp(std::optional<int> a, std::optional<double> b) : a(a.value_or(3)), b(b.value_or(4.0)){}
	void print()
	{
		std::cout << a << " " << b << std::endl;
	}
	int a;
	double b;
};

int main()
{
	temp t(std::nullopt, 8.0);
	temp t2(10, std::nullopt);
	t.print();
	t2.print();
	std::optional<int> oint;
	std::cout << "oint: " << oint.value_or(3) << "\n";
	return 0;
}
