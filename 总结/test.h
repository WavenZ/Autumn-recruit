#pragma once
#include <bits/stdc++.h>
using namespace std;
struct Node{
    Node(int value) : val(value), next(nullptr) { }
    int val;
    Node* next;
};
Node* make_list(int* vec, int begin, int end){
    if(begin > end) return nullptr;
    Node* head = new Node(vec[end]);
    for(int i = end - 1; i >= begin; --i){
        Node* curr = new Node(vec[i]);
        curr->next = head;
        head = curr;
    }
    return head;
}
Node* list_reverse(Node* head){
    if(head == nullptr || head->next == nullptr) return head;
    Node* prev = nullptr, *curr = head, *next = head->next;
    while(curr){
        curr->next = prev;
        prev = curr;
        curr = next;
        if(next) next = next->next;
    }
    return prev;
}
Node* list_merge(Node* l1, Node* l2){
    if(l1 == nullptr) return l2;
    if(l2 == nullptr) return l1;
    if(l1->val <= l2->val){
        l1->next = list_merge(l1->next, l2);
        return l1;
    }else{
        l2->next = list_merge(l1, l2->next);
        return l2;
    }
    return l1;
}
void traverse_list(Node* head){
    Node* curr = head;
    while(curr){
        cout << curr->val << " ";
        curr = curr->next;
    }
    cout << endl;
}

struct TreeNode{
    TreeNode(int value) : val(value) { }
    int val;
    TreeNode* left, *right;
};