namespace RMV.ML.MNIST.Demo;

   partial class View
   {
      /// <summary>
      /// Required designer variable.
      /// </summary>
      private System.ComponentModel.IContainer components = null;

      /// <summary>
      /// Clean up any resources being used.
      /// </summary>
      /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
      protected override void Dispose( bool disposing )
      {
         if( disposing && ( components != null ) )
         {
            components.Dispose();
         }
         base.Dispose( disposing );
      }

	#region Windows Form Designer generated code

	/// <summary>
	/// Required method for Designer support - do not modify
	/// the contents of this method with the code editor.
	/// </summary>
	private void InitializeComponent()
	{
		statusStrip1 = new StatusStrip();
		menuStrip1 = new MenuStrip();
		splitContainer1 = new SplitContainer();
		pictureBox = new PictureBox();
		textPredict = new TextBox();
		label2 = new Label();
		label1 = new Label();
		textActual = new TextBox();
		SelectImage = new NumericUpDown();
		( ( System.ComponentModel.ISupportInitialize )splitContainer1 ).BeginInit();
		splitContainer1.Panel1.SuspendLayout();
		splitContainer1.Panel2.SuspendLayout();
		splitContainer1.SuspendLayout();
		( ( System.ComponentModel.ISupportInitialize )pictureBox ).BeginInit();
		( ( System.ComponentModel.ISupportInitialize )SelectImage ).BeginInit();
		SuspendLayout();
		// 
		// statusStrip1
		// 
		statusStrip1.Location = new Point( 0, 443 );
		statusStrip1.Name = "statusStrip1";
		statusStrip1.Padding = new Padding( 1, 0, 16, 0 );
		statusStrip1.Size = new Size( 654, 22 );
		statusStrip1.TabIndex = 0;
		statusStrip1.Text = "statusStrip1";
		// 
		// menuStrip1
		// 
		menuStrip1.Location = new Point( 0, 0 );
		menuStrip1.Name = "menuStrip1";
		menuStrip1.Padding = new Padding( 7, 2, 0, 2 );
		menuStrip1.Size = new Size( 654, 24 );
		menuStrip1.TabIndex = 1;
		menuStrip1.Text = "menuStrip1";
		// 
		// splitContainer1
		// 
		splitContainer1.Dock = DockStyle.Fill;
		splitContainer1.Location = new Point( 0, 24 );
		splitContainer1.Margin = new Padding( 4, 3, 4, 3 );
		splitContainer1.Name = "splitContainer1";
		// 
		// splitContainer1.Panel1
		// 
		splitContainer1.Panel1.Controls.Add( pictureBox );
		// 
		// splitContainer1.Panel2
		// 
		splitContainer1.Panel2.Controls.Add( textPredict );
		splitContainer1.Panel2.Controls.Add( label2 );
		splitContainer1.Panel2.Controls.Add( label1 );
		splitContainer1.Panel2.Controls.Add( textActual );
		splitContainer1.Panel2.Controls.Add( SelectImage );
		splitContainer1.Panel2.Font = new Font( "Microsoft Sans Serif", 9F, FontStyle.Regular, GraphicsUnit.Point,   0 );
		splitContainer1.Size = new Size( 654, 419 );
		splitContainer1.SplitterDistance = 473;
		splitContainer1.SplitterWidth = 5;
		splitContainer1.TabIndex = 2;
		// 
		// pictureBox
		// 
		pictureBox.Dock = DockStyle.Fill;
		pictureBox.Location = new Point( 0, 0 );
		pictureBox.Margin = new Padding( 4, 3, 4, 3 );
		pictureBox.Name = "pictureBox";
		pictureBox.Size = new Size( 473, 419 );
		pictureBox.TabIndex = 0;
		pictureBox.TabStop = false;
		// 
		// textPredict
		// 
		textPredict.Font = new Font( "Microsoft Sans Serif", 9F, FontStyle.Bold, GraphicsUnit.Point,   0 );
		textPredict.Location = new Point( 113, 116 );
		textPredict.Name = "textPredict";
		textPredict.Size = new Size( 49, 21 );
		textPredict.TabIndex = 6;
		// 
		// label2
		// 
		label2.AutoSize = true;
		label2.Location = new Point( 31, 116 );
		label2.Name = "label2";
		label2.Size = new Size( 59, 15 );
		label2.TabIndex = 5;
		label2.Text = "Predicted";
		// 
		// label1
		// 
		label1.AutoSize = true;
		label1.Location = new Point( 50, 89 );
		label1.Margin = new Padding( 4, 0, 4, 0 );
		label1.Name = "label1";
		label1.Size = new Size( 40, 15 );
		label1.TabIndex = 4;
		label1.Text = "Actual";
		// 
		// textActual
		// 
		textActual.Font = new Font( "Microsoft Sans Serif", 9F, FontStyle.Bold, GraphicsUnit.Point,   0 );
		textActual.Location = new Point( 111, 84 );
		textActual.Margin = new Padding( 4, 3, 4, 3 );
		textActual.Name = "textActual";
		textActual.Size = new Size( 51, 21 );
		textActual.TabIndex = 3;
		// 
		// SelectImage
		// 
		SelectImage.Font = new Font( "Microsoft Sans Serif", 9F, FontStyle.Bold, GraphicsUnit.Point,   0 );
		SelectImage.Location = new Point( 73, 29 );
		SelectImage.Margin = new Padding( 4, 3, 4, 3 );
		SelectImage.Maximum = new decimal( new int[] { 10000, 0, 0, 0 } );
		SelectImage.Minimum = new decimal( new int[] { 1, 0, 0, 0 } );
		SelectImage.Name = "SelectImage";
		SelectImage.Size = new Size( 89, 21 );
		SelectImage.TabIndex = 2;
		SelectImage.Value = new decimal( new int[] { 1, 0, 0, 0 } );
		SelectImage.ValueChanged +=  SlectImage_Click ;
		// 
		// View
		// 
		AutoScaleDimensions = new SizeF( 7F, 15F );
		AutoScaleMode = AutoScaleMode.Font;
		ClientSize = new Size( 654, 465 );
		Controls.Add( splitContainer1 );
		Controls.Add( statusStrip1 );
		Controls.Add( menuStrip1 );
		MainMenuStrip = menuStrip1;
		Margin = new Padding( 4, 3, 4, 3 );
		MaximizeBox = false;
		Name = "View";
		StartPosition = FormStartPosition.CenterScreen;
		Text = "Classification Demo";
		splitContainer1.Panel1.ResumeLayout( false );
		splitContainer1.Panel2.ResumeLayout( false );
		splitContainer1.Panel2.PerformLayout();
		( ( System.ComponentModel.ISupportInitialize )splitContainer1 ).EndInit();
		splitContainer1.ResumeLayout( false );
		( ( System.ComponentModel.ISupportInitialize )pictureBox ).EndInit();
		( ( System.ComponentModel.ISupportInitialize )SelectImage ).EndInit();
		ResumeLayout( false );
		PerformLayout();

	}

	#endregion

	private System.Windows.Forms.StatusStrip statusStrip1;
      private System.Windows.Forms.MenuStrip menuStrip1;
      private System.Windows.Forms.SplitContainer splitContainer1;
      private System.Windows.Forms.PictureBox pictureBox;
      private System.Windows.Forms.NumericUpDown SelectImage;
      private System.Windows.Forms.Label label1;
      private System.Windows.Forms.TextBox textActual;
	private TextBox textPredict;
	private Label label2;
}
