using RMV.ML.Network.Common;

namespace RMV.ML.MNIST.CS;

/// <summary>
/// Multilayer neural network training for the MNIST dataset using various optimization algorithms.
/// </summary>
class Program
{
    public static void Main()
    {
		var settings = ConfigManager.GetRoot<AppSettings>() ?? throw new Exception( "Failed to load configuration" );
		
		var parser = new Parser( settings.Output );
		
		var trainSet = parser.Run( File.ReadAllLines( settings.TrainPath ) );
		var testSet = parser.Run( File.ReadAllLines( settings.TestPath ) );

		Controller controller = new( trainSet, testSet, settings );
		controller.RunML(  );
    }
}