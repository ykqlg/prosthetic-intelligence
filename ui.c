#include <gtk/gtk.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// 定义全局变量，用于存储数据文件名
char dataFileName[256];
FILE *file = NULL;
GtkWidget *textEntry;
char labelFilePath[] = "./I2C/data/velcro_label.csv"; // 全局变量保存文件路径

void runDataCollector()
{
    printf("\nStart Collection\n");

    system("./main_index");
    FILE *file = fopen("targetFileName.txt", "r");
    if (file == NULL)
    {
        perror("Error opening file");
    }
    fgets(dataFileName, sizeof(dataFileName), file);

    // printf("\n%s\n",dataFileName);
    fclose(file);
}

void runDataVisualization()
{
    system("python3 visualization.py");
}

void runPrediction()
{
    // printf("\n-->Prediction:\n");

    // system("./test_diff_object.sh");
    // printf("Finished!\n");

}

void onOKClicked(GtkWidget *widget, gpointer data)
{
    const gchar *text = gtk_entry_get_text(GTK_ENTRY(textEntry));
    if (strcmp(text, "0") == 0 || strcmp(text, "1") == 0)
    {
        // 打开附加标签文件，追加文件名和标签信息
        FILE *labelFile = fopen(labelFilePath, "a");
        if (labelFile != NULL)
        {
            // 检查文件是否为空
            fseek(labelFile, 0, SEEK_END);
            int isEmpty = ftell(labelFile) == 0;
            fseek(labelFile, 0, SEEK_SET);

            // 如果文件为空，写入列名
            if (isEmpty)
            {
                fprintf(labelFile, "FilePath,Label\n");
            }

            fprintf(labelFile, "%s,%s\n", dataFileName, text);
            fclose(labelFile);
        }
    }
    runPrediction();

    gtk_main_quit();
}

void onContinueClicked(GtkWidget *widget, gpointer data)
{
    const gchar *text = gtk_entry_get_text(GTK_ENTRY(textEntry));
    if (strcmp(text, "0") == 0 || strcmp(text, "1") == 0)
    {
        // 打开附加标签文件，追加文件名和标签信息
        FILE *labelFile = fopen(labelFilePath, "a");
        if (labelFile != NULL)
        {
            fprintf(labelFile, "%s,%s\n", dataFileName, text);
            fclose(labelFile);
        }
    }
    runPrediction();

    // 关闭当前窗口
    GtkWidget *window = GTK_WIDGET(data);
    gtk_widget_destroy(window);
    execlp("./ui", "ui", NULL);
}

void onCancelClicked(GtkWidget *widget, gpointer data)
{
    // 在这里处理点击"Cancel"按钮的逻辑
    gtk_main_quit();
    if (remove(dataFileName) == 0)
    {
        printf("File %s deleted successfully.\n", dataFileName);
    }
    else
    {
        perror("Error deleting file");
    }
}

void onRestartClicked(GtkWidget *widget, gpointer data)
{
    if (remove(dataFileName) == 0)
    {
        printf("File %s deleted successfully.\n", dataFileName);
    }
    else
    {
        perror("Error deleting file");
    }
    // 关闭当前窗口
    GtkWidget *window = GTK_WIDGET(data);
    if (GTK_IS_WIDGET(window))
    {
        gtk_widget_destroy(window);
        execlp("./ui", "ui", NULL);
    }
    else
    {
        g_print("Invalid GtkWidget pointer\n");
    }
}

// 主函数
int main(int argc, char *argv[])
{
    // 运行C语言程序，将数据写入到一个csv文件中
    runDataCollector();

    // python脚本对那个csv文件数据进行可视化，产生图像文件visualization.png
    runDataVisualization();

    GtkWidget *window, *buttonOK, *buttonCancel, *buttonContinue, *buttonRestart, *image, *box;
    gtk_init(&argc, &argv);

    // 创建顶层窗口
    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "Data Viewer");
    gtk_window_set_default_size(GTK_WINDOW(window), 200, 200);

    // 创建"OK"按钮
    buttonOK = gtk_button_new_with_label("OK & Cancel");
    g_signal_connect(buttonOK, "clicked", G_CALLBACK(onOKClicked), window);

    // 创建"Continue"按钮
    buttonContinue = gtk_button_new_with_label("OK & Continue");
    g_signal_connect(buttonContinue, "clicked", G_CALLBACK(onContinueClicked), window);

    // 创建"Cancel"按钮
    buttonCancel = gtk_button_new_with_label("Cancel & Quit");
    g_signal_connect(buttonCancel, "clicked", G_CALLBACK(onCancelClicked), window);

    // 创建"Restart"按钮
    buttonRestart = gtk_button_new_with_label("Restart");
    g_signal_connect(buttonRestart, "clicked", G_CALLBACK(onRestartClicked), window);

    // 创建输入文本框
    textEntry = gtk_entry_new();

    // 创建图形显示区域
    image = gtk_image_new_from_file("visualization.png");

    // 创建布局
    box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_box_pack_start(GTK_BOX(box), image, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(box), textEntry, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(box), buttonOK, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(box), buttonContinue, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(box), buttonCancel, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(box), buttonRestart, FALSE, FALSE, 0);

    gtk_container_add(GTK_CONTAINER(window), box);

    gtk_widget_show_all(window);

    gtk_main();

    return 0;
}
