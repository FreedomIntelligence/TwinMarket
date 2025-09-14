"""
论坛数据库管理和社交互动模块

该模块负责管理论坛系统的所有数据库操作和社交互动功能，包括帖子管理、
用户互动、内容推荐等核心功能。是整个社交交易系统的论坛数据管理核心。

核心功能：
- 论坛数据库管理：帖子、反应、引用关系的完整数据管理
- 社交互动处理：点赞、取消点赞、转发等用户互动行为
- 内容推荐系统：基于社交网络和热度算法的帖子推荐
- 评分系统：动态的帖子评分和排序机制
- 异步操作支持：高性能的异步数据库操作

技术特性：
- 基于SQLite的高性能数据存储
- 异步数据库操作支持（aiosqlite）
- 复杂的评分算法（热度算法、PageRank等）
- 缓存优化的查询性能
- 完整的数据一致性保证

数据模型：
- posts表：帖子内容、评分、类型、时间戳
- reactions表：用户互动行为（点赞、取消点赞、转发）
- post_references表：帖子间的引用关系

适用场景：
- 社交交易平台
- 投资观点分享
- 用户互动分析
- 内容推荐系统
- 社交影响力分析
"""

# 标准库导入
import math
import os
import sqlite3
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

# 第三方库导入
import aiosqlite
import networkx as nx
import pandas as pd

# ============================ 全局配置 ============================

# 默认论坛数据库路径
FORUM_DB_PATH = 'data/ForumDB/syn_100.db'
# 默认用户数据库路径  
USER_DB_PATH = 'data/UserDB/sys_100.db'


def init_db_forum(db_path=FORUM_DB_PATH):
    """
    初始化论坛数据库 - 创建完整的论坛数据结构
    
    该函数负责创建论坛系统所需的所有数据表和索引，支持完整的
    社交互动功能。如果数据库已存在，会先删除后重新创建。
    
    数据表结构：
    1. posts表：存储帖子内容、评分、类型等核心信息
    2. reactions表：存储用户互动行为（点赞、取消点赞、转发）
    3. post_references表：存储帖子间的引用关系
    
    索引优化：
    - posts表：user_id索引（提高用户帖子查询性能）
    - reactions表：post_id索引（提高帖子互动查询性能）
    - post_references表：reference_id索引（提高引用关系查询性能）
    
    Args:
        db_path (str): SQLite数据库文件路径，默认使用全局FORUM_DB_PATH
        
    Note:
        - 会删除现有数据库文件（如果存在）
        - 自动创建必要的目录结构
        - 包含完整的外键约束和数据完整性检查
        - 支持帖子类型和反应类型的枚举约束
    """
    # If the database file already exists, delete it
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Deleted existing database file: {db_path}")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        # Create posts table
        conn.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            score INTEGER DEFAULT 0,
            belief TEXT,
            type TEXT,    -- New column for post type
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')

        # Create reactions table
        conn.execute('''
        CREATE TABLE IF NOT EXISTS reactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            post_id INTEGER NOT NULL,
            type TEXT CHECK(type IN ('repost', 'like', 'unlike')) NOT NULL,  -- New column for reaction type
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (post_id) REFERENCES posts(id)
        )''')

        # Create post_references table
        conn.execute('''
        CREATE TABLE IF NOT EXISTS post_references (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reference_id INTEGER NOT NULL,  -- ID of the referenced post
            repost_id INTEGER NOT NULL,     -- ID of the repost
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (reference_id) REFERENCES posts(id),
            FOREIGN KEY (repost_id) REFERENCES posts(id),
            UNIQUE(reference_id, repost_id)  -- Ensure each reference_id and repost_id combination is unique
        )''')

        # Create indexes
        conn.execute('CREATE INDEX IF NOT EXISTS idx_posts_user_id ON posts(user_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_reactions_post_id ON reactions(post_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_post_references_ref_id ON post_references(reference_id)')

        conn.commit()
        print(f"Initialized new database file: {db_path}")


def update_posts_score_by_date(
    target_date: str,
    db_path: str = FORUM_DB_PATH
) -> bool:
    """
    Update the score of posts based on reactions for a specific date.

    Args:
        target_date: The date for which to calculate and update scores (format: 'YYYY-MM-DD').
        db_path: Path to the SQLite database.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("BEGIN")

            # Calculate the total score for each post based on reactions for the target date
            # Assign scores as follows:
            # - 'like' = +1
            # - 'unlike' = -1
            # - 'repost' = +0 (no impact on score)
            cursor = conn.execute('''
                SELECT post_id,
                       SUM(CASE WHEN type = 'like' THEN 1
                               WHEN type = 'unlike' THEN -1
                               ELSE 0 END) as total_score
                FROM reactions
                WHERE DATE(created_at) = ?
                GROUP BY post_id
            ''', (target_date,))

            # Fetch all results
            post_scores = cursor.fetchall()

            # Update the posts table with the calculated scores
            for post_id, total_score in post_scores:
                conn.execute('''
                    UPDATE posts
                    SET score = ?
                    WHERE id = ?
                ''', (total_score, post_id))

            conn.commit()
            return True

    except sqlite3.Error as e:
        print(f"Error updating posts score: {e}")
        conn.rollback()  # Rollback in case of error
        return False


def update_posts_score_by_date_range(
    start_date: str = '2023-01-01',
    end_date: str = '2023-06-16',
    db_path: str = "FORUM_DB_PATH"
) -> tuple[bool, pd.DataFrame]:
    """
    Update the score of posts based on reactions for a specific date range.

    Args:
        start_date: The start date of the range (format: 'YYYY-MM-DD').
        end_date: The end date of the range (format: 'YYYY-MM-DD').
        db_path: Path to the SQLite database.

    Returns:
        tuple: (success: bool, scores_df: pd.DataFrame)
            - success: True if update was successful
            - scores_df: DataFrame containing post IDs and their updated scores
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("BEGIN")

            # Reset scores to 0 for posts within the date range
            conn.execute('''
                UPDATE posts 
                SET score = 0 
                WHERE DATE(created_at) BETWEEN ? AND ?
            ''', (start_date, end_date))

            # Calculate the total score for each post based on reactions within the date range
            cursor = conn.execute('''
                SELECT post_id,
                       SUM(CASE WHEN type = 'like' THEN 1
                               WHEN type = 'unlike' THEN -1
                               ELSE 0 END) as total_score
                FROM reactions
                WHERE DATE(created_at) BETWEEN ? AND ?
                GROUP BY post_id
            ''', (start_date, end_date))

            # Fetch all results
            post_scores = cursor.fetchall()

            # Update the posts table with the calculated scores
            for post_id, total_score in post_scores:
                cursor.execute('''
                    SELECT type FROM posts WHERE id = ?
                ''', (post_id,))
                post_type_row = cursor.fetchone()

                if post_type_row and post_type_row[0] == 'repost':
                    root_post_id = find_root_post(post_id, db_path)
                    if root_post_id:
                        path = root_post_id['path']
                        path.append(root_post_id['post_id'])
                        for id in root_post_id['path']:
                            conn.execute('''
                                UPDATE posts
                                SET score = score + ?
                                WHERE id = ?
                            ''', (total_score, id))
                else:
                    conn.execute('''
                        UPDATE posts
                        SET score = ?
                        WHERE id = ?
                    ''', (total_score, post_id))

            # Get final scores for all posts in date range
            scores_df = pd.read_sql_query('''
                SELECT id as post_id, score
                FROM posts 
            ''', conn)

            conn.commit()
            return True, scores_df

    except sqlite3.Error as e:
        print(f"Error updating posts score: {e}")
        conn.rollback()
        return False, pd.DataFrame()
    

def fetch_posts_score_by_date_range(
    start_date: str = '2023-01-01',
    end_date: str = '2023-06-16',
    db_path: str = "FORUM_DB_PATH"
) -> pd.DataFrame:
    """
    Fetch cumulative post scores based on reactions for a specific date range.

    Args:
        start_date: The start date of the range (format: 'YYYY-MM-DD').
        end_date: The end date of the range (format: 'YYYY-MM-DD').
        db_path: Path to the SQLite database.

    Returns:
        pd.DataFrame: DataFrame containing post IDs and their cumulative scores.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            # 获取所有帖子
            posts_df = pd.read_sql_query('''
                SELECT id AS post_id, type
                FROM posts
                WHERE DATE(created_at) BETWEEN ? AND ?
            ''', conn, params=(start_date, end_date))

            # 获取点赞和取消点赞数据
            reactions_df = pd.read_sql_query('''
                SELECT post_id,
                       SUM(CASE WHEN type = 'like' THEN 1
                               WHEN type = 'unlike' THEN -1
                               ELSE 0 END) as total_score
                FROM reactions
                WHERE DATE(created_at) BETWEEN ? AND ?
                GROUP BY post_id
            ''', conn, params=(start_date, end_date))

        # 合并帖子信息和 reactions 统计数据
        scores_df = posts_df.merge(reactions_df, on="post_id", how="left").fillna(0)

        # 处理 repost 级联加分
        for index, row in scores_df.iterrows():
            if row['type'] == 'repost':
                root_post_id = find_root_post(row['post_id'], db_path)
                if root_post_id:
                    path = root_post_id['path']
                    path.append(root_post_id['post_id'])
                    for post_id in path:
                        scores_df.loc[scores_df['post_id'] == post_id, 'total_score'] += row['total_score']

        # 仅保留最终需要的列
        scores_df = scores_df[['post_id', 'total_score']]

        return scores_df

    except sqlite3.Error as e:
        print(f"Error fetching posts score: {e}")
        return pd.DataFrame()




def get_user_net_likes_and_post_interactions(
    user_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    db_path: str = FORUM_DB_PATH
) -> Tuple[int, pd.DataFrame]:
    """
    获取用户在指定时间范围内的净点赞数量及其发布的帖子的互动详情。

    Args:
        user_id (str): 用户的ID。
        start_date (pd.Timestamp): 开始时间。
        end_date (pd.Timestamp): 结束时间。
        db_path (str): 数据库路径。默认为 FORUM_DB_PATH。

    Returns:
        Tuple[int, pd.DataFrame]:
            - int: 用户获得的净点赞数量（总点赞数 - 总取消点赞数）。
            - pd.DataFrame: 包含用户发布的帖子的互动详情，列包括：
                - post_id: 帖子ID
                - created_at: 帖子发布时间
                - like_users: 点赞的用户列表
                - unlike_users: 取消点赞的用户列表
                - repost_users: 转发的用户列表
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row

            # 1. 计算用户获得的净点赞数量
            query_net_likes = """
                SELECT 
                    SUM(CASE WHEN r.type = 'like' THEN 1 ELSE 0 END) as total_likes,
                    SUM(CASE WHEN r.type = 'unlike' THEN 1 ELSE 0 END) as total_unlikes
                FROM reactions r
                JOIN posts p ON r.post_id = p.id
                WHERE p.user_id = ?
                AND r.created_at >= ? AND r.created_at <= ?
            """
            cursor = conn.execute(query_net_likes, (user_id, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
            result = cursor.fetchone()
            total_likes = result["total_likes"] if result["total_likes"] else 0
            total_unlikes = result["total_unlikes"] if result["total_unlikes"] else 0
            net_likes = total_likes - total_unlikes

            # 2. 获取用户发布的帖子的互动详情
            query_post_interactions = """
                SELECT 
                    p.id as post_id,
                    p.created_at,
                    GROUP_CONCAT(DISTINCT CASE WHEN r.type = 'like' THEN r.user_id END) as like_users,
                    GROUP_CONCAT(DISTINCT CASE WHEN r.type = 'unlike' THEN r.user_id END) as unlike_users,
                    GROUP_CONCAT(DISTINCT CASE WHEN r.type = 'repost' THEN r.user_id END) as repost_users
                FROM posts p
                LEFT JOIN reactions r ON p.id = r.post_id
                WHERE p.user_id = ?
                AND p.created_at >= ? AND p.created_at <= ?
                GROUP BY p.id
            """
            cursor = conn.execute(query_post_interactions, (user_id, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
            rows = cursor.fetchall()

            # 将结果转换为 DataFrame
            data = []
            for row in rows:
                data.append({
                    "post_id": row["post_id"],
                    "created_at": row["created_at"],
                    "like_users": row["like_users"].split(",") if row["like_users"] else [],
                    "unlike_users": row["unlike_users"].split(",") if row["unlike_users"] else [],
                    "repost_users": row["repost_users"].split(",") if row["repost_users"] else []
                })
            df = pd.DataFrame(data)

            return net_likes, df

    except sqlite3.Error as e:
        print(f"Error getting user net likes and post interactions: {e}")
        return 0, pd.DataFrame()


def create_post_db(
    user_id: str,
    content: str,
    belief: Optional[str] = None,
    type: Optional[str] = None,
    created_at: Optional[pd.Timestamp] = None,
    db_path: str = FORUM_DB_PATH
) -> Optional[int]:
    """
    在论坛数据库中创建新帖子
    
    该函数负责在论坛系统中创建新的帖子记录，支持多种帖子类型
    和用户信念值的记录，是用户发布内容的核心接口。
    
    帖子信息包含：
    - 基础信息：用户ID、内容、创建时间
    - 扩展信息：用户信念值、帖子类型
    - 系统信息：自动生成的帖子ID、初始评分
    
    Args:
        user_id (str): 发帖用户的ID
        content (str): 帖子的文本内容
        belief (Optional[str]): 用户的投资信念值，可选
        type (Optional[str]): 帖子类型，可选
        created_at (Optional[pd.Timestamp]): 帖子创建时间，默认当前时间
        db_path (str): 数据库文件路径，默认使用全局FORUM_DB_PATH
        
    Returns:
        Optional[int]: 创建成功返回帖子ID，失败返回None
        
    Note:
        - 帖子ID由数据库自动生成
        - 初始评分为0，后续通过互动更新
        - 支持时间戳的精确记录
        - 包含完整的错误处理机制
    """
    try:
        with sqlite3.connect(db_path) as conn:
            if created_at is None:
                created_at = pd.Timestamp.now()

            cursor = conn.execute('''
                INSERT INTO posts (user_id, content, belief, type, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, content, belief, type, created_at.strftime('%Y-%m-%d %H:%M:%S')))

            post_id = cursor.lastrowid
            conn.commit()
            return post_id

    except sqlite3.Error as e:
        print(f"Error creating post: {e}")
        return None


def repost_db(
    reference_id: str,
    user_id: str,
    content: str,
    belief: Optional[str] = None,
    created_at: Optional[pd.Timestamp] = None,
    db_path: str = FORUM_DB_PATH
) -> Optional[int]:
    """
    Create a repost referencing another post and record a reaction of type 'repost'.

    Args:
        reference_id (str): ID of the post being referenced.
        user_id (str): ID of user making the repost.
        content (str): Content of the repost.
        belief (Optional[str]): The belief of the user (default: None).
        created_at (Optional[pd.Timestamp]): Timestamp for the post. Defaults to current time.
        db_path (str): Path to database.

    Returns:
        Optional[int]: ID of created repost, None if failed.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            # Start transaction
            conn.execute("BEGIN")

            # Verify reference exists
            cursor = conn.execute('SELECT 1 FROM posts WHERE id = ?', (reference_id,))
            if not cursor.fetchone():
                print(f"Reference post {reference_id} not found")
                return None

            if created_at is None:
                created_at = pd.Timestamp.now()

            # Create the repost with type set to 'repost'
            cursor = conn.execute('''
                INSERT INTO posts (user_id, content, belief, type, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, content, belief, 'repost', created_at.strftime('%Y-%m-%d %H:%M:%S')))

            new_post_id = cursor.lastrowid

            # Create reference linking the original post and the new repost
            conn.execute('''
                INSERT INTO post_references (reference_id, repost_id, created_at)
                VALUES (?, ?, ?)
            ''', (reference_id, new_post_id, created_at.strftime('%Y-%m-%d %H:%M:%S')))

            # Record a reaction of type 'repost'
            conn.execute('''
                INSERT INTO reactions (user_id, post_id, type, created_at)
                VALUES (?, ?, ?, ?)
            ''', (user_id, reference_id, 'repost', created_at.strftime('%Y-%m-%d %H:%M:%S')))

            conn.commit()
            return new_post_id

    except sqlite3.Error as e:
        print(f"Error creating repost: {e}")
        conn.rollback()  # Rollback in case of error
        return None


async def like_post_db(
    user_id: str,
    post_id: int,
    created_at: Optional[pd.Timestamp] = None,
    db_path: str = FORUM_DB_PATH
) -> bool:
    """Like a post (type 'like') - Only updates reactions table."""
    try:
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("BEGIN")

            # Check if post exists
            cursor = await conn.execute('SELECT id FROM posts WHERE id = ?', (post_id,))
            post = await cursor.fetchone()
            if not post:
                print(f"Post {post_id} not found")
                return False

            # Get current reaction if exists
            cursor = await conn.execute(
                'SELECT type FROM reactions WHERE user_id = ? AND post_id = ?',
                (user_id, post_id)
            )
            reaction = await cursor.fetchone()

            if created_at is None:
                created_at = pd.Timestamp.now()

            if reaction is None:
                # New reaction
                await conn.execute('''
                    INSERT INTO reactions (user_id, post_id, type, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, post_id, 'like', created_at.strftime('%Y-%m-%d %H:%M:%S')))
            else:
                # Update existing reaction
                old_type = reaction[0]
                if old_type != 'like':  # Only update if not already liked
                    await conn.execute('''
                        UPDATE reactions 
                        SET type = ?, created_at = ?
                        WHERE user_id = ? AND post_id = ?
                    ''', ('like', created_at.strftime('%Y-%m-%d %H:%M:%S'), user_id, post_id))

            await conn.commit()
            return True

    except aiosqlite.Error as e:
        print(f"Error liking post: {e}")
        return False


async def unlike_post_db(
    user_id: str,
    post_id: int,
    created_at: Optional[pd.Timestamp] = None,
    db_path: str = FORUM_DB_PATH
) -> bool:
    """Unlike a post (type 'unlike') - Only updates reactions table."""
    try:
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("BEGIN")

            # Check if post exists
            cursor = await conn.execute('SELECT id FROM posts WHERE id = ?', (post_id,))
            post = await cursor.fetchone()
            if not post:
                print(f"Post {post_id} not found")
                return False

            # Get current reaction if exists
            cursor = await conn.execute(
                'SELECT type FROM reactions WHERE user_id = ? AND post_id = ?',
                (user_id, post_id)
            )
            reaction = await cursor.fetchone()

            if created_at is None:
                created_at = pd.Timestamp.now()

            if reaction is None:
                # New reaction
                await conn.execute('''
                    INSERT INTO reactions (user_id, post_id, type, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, post_id, 'unlike', created_at.strftime('%Y-%m-%d %H:%M:%S')))
            else:
                # Update existing reaction
                old_type = reaction[0]
                if old_type != 'unlike':  # Only update if not already unliked
                    await conn.execute('''
                        UPDATE reactions 
                        SET type = ?, created_at = ?
                        WHERE user_id = ? AND post_id = ?
                    ''', ('unlike', created_at.strftime('%Y-%m-%d %H:%M:%S'), user_id, post_id))

            await conn.commit()
            return True

    except aiosqlite.Error as e:
        print(f"Error unliking post: {e}")
        return False


async def execute_forum_actions(
    forum_args: List[Dict],
    user_id: str,
    db_path: str,
    belief: Optional[str] = None,
    created_at: Optional[pd.Timestamp] = None,
) -> None:
    """
    Execute forum actions based on the provided action list in forum_args.
    """
    if created_at is None:
        created_at = pd.Timestamp.now()

    for action in forum_args:
        action_type = action.get("action")
        post_id = action.get("post_id")
        content = action.get("reason", "")

        if action_type == "repost":
            # Repost action
            if not content:
                continue  # Skip if content is missing

            repost_db(
                reference_id=int(post_id),
                user_id=user_id,
                content=content,
                belief=belief,
                created_at=created_at,
                db_path=db_path
            )

        elif action_type == "like":
            # Like action
            await like_post_db(
                user_id=user_id,
                post_id=post_id,
                created_at=created_at,
                db_path=db_path
            )

        elif action_type == "unlike":
            # Unlike action
            await unlike_post_db(
                user_id=user_id,
                post_id=post_id,
                created_at=created_at,
                db_path=db_path
            )


def get_all_users_posts_db(
    end_date: Optional[pd.Timestamp] = None,
    db_path: str = FORUM_DB_PATH
) -> Dict[str, List[Dict]]:
    """
    Get all posts by all users within a date range, including all post-related attributes.

    Args:
        end_date (Optional[pd.Timestamp]): End of the date range (inclusive).
        db_path (str): Path to the database. Defaults to FORUM_DB_PATH.

    Returns:
        dict: A dictionary where the key is the user ID and the value is a list of post dictionaries.
              Each post dictionary contains:
                - id: Post ID
                - user_id: User ID of the poster
                - content: Post content
                - score: Current post score
                - belief: Belief of the user
                - type: Type of the post
                - created_at: Timestamp of the post
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()

            # Base query to fetch posts for all users
            query = """
                SELECT 
                    id,
                    user_id,
                    content,
                    score,
                    belief,
                    type,
                    created_at
                FROM posts
                WHERE type != 'repost'
            """
            params = []

            # Add date range filter if end_date is provided
            if end_date:
                query += " AND created_at <= ? ORDER BY created_at DESC"
                params.append(end_date.strftime('%Y-%m-%d %H:%M:%S'))
            else:
                query += " ORDER BY created_at DESC"
                
            # Execute the query
            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Organize posts by user_id
            user_posts = {}
            for row in rows:
                post = dict(row)
                user_id = post["user_id"]

                if user_id not in user_posts:
                    user_posts[user_id] = []
                user_posts[user_id].append(post)

            return user_posts

    except sqlite3.Error as e:
        print(f"Error getting all users' posts: {e}")
        return {}

def get_user_posts_db(
    user_id: str,
    end_date: Optional[pd.Timestamp] = None,
    db_path: str = FORUM_DB_PATH
) -> List[Dict]:
    """
    Get all posts by a specific user within a date range, including all post-related attributes.

    Args:
        user_id (str): The ID of the user whose posts are being retrieved.
        end_date (Optional[pd.Timestamp]): End of the date range (inclusive).
        db_path (str): Path to the database. Defaults to FORUM_DB_PATH.

    Returns:
        list[dict]: A list of post dictionaries, each containing:
            - id: Post ID
            - user_id: User ID of the poster
            - content: Post content
            - score: Current post score
            - belief: Belief of the user
            - type: Type of the post
            - created_at: Timestamp of the post
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()

            # Base query to fetch posts by the user
            query = """
                SELECT 
                    id,
                    user_id,
                    content,
                    score,
                    belief,
                    type,
                    created_at
                FROM posts
                WHERE user_id = ?
            """
            params = [user_id]

            # Add date range filter if end_date is provided
            if end_date:
                query += " AND created_at <= ?"
                params.append(end_date.strftime('%Y-%m-%d %H:%M:%S'))

            # Execute the query
            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Convert rows to a list of dictionaries
            posts = [dict(row) for row in rows]

            return posts

    except sqlite3.Error as e:
        print(f"Error getting user posts: {e}")
        return []


@lru_cache(maxsize=128)
def compute_pagerank(graph: nx.Graph) -> Dict:
    """Cache PageRank calculations as they are computationally expensive."""
    return nx.pagerank(graph, weight="weight")


def get_cached_user_posts(
    user_id: str,
    db_path: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> List[Dict]:
    """Cache user posts with TTL."""
    return get_user_posts_db(
        user_id=user_id,
        db_path=db_path,
        start_date=start_date,
        end_date=end_date
    )


# def recommend_posts(
#     graph: nx.Graph,
#     target_user_id: str,
#     db_path: str,
#     start_date: pd.Timestamp,
#     end_date: pd.Timestamp,
#     max_return: int = 3
# ) -> List[Dict]:
#     """
#     Recommend posts for a target user using PageRank and post data.
#     Now with caching support for better performance.
#     """
#     # Step 1: Compute PageRank scores (cached)
#     pagerank_scores = compute_pagerank(graph)

#     # Step 2: Retrieve posts from all users except target user (cached)
#     all_posts = []
#     for user_id in graph.nodes():
#         if user_id != target_user_id:
#             posts = get_cached_user_posts(
#                 user_id=user_id,
#                 db_path=db_path,
#                 start_date=start_date,
#                 end_date=end_date
#             )
#             all_posts.extend(posts)

#     # Step 3: Rank posts based on PageRank, like_score, and timestamp
#     ranked_posts = []
#     for post in all_posts:
#         post_score = (
#             pagerank_scores.get(post["user_id"], 0) * 0.5 +
#             post["like_score"] * 0.3 +
#             (pd.to_datetime(post["created_at"]).timestamp() / 1e9) * 0.2
#         )
#         ranked_posts.append((post_score, post))

#     # Sort posts by score in descending order
#     ranked_posts.sort(key=lambda x: x[0], reverse=True)

#     # Step 4: Return the top posts
#     return [post for _, post in ranked_posts[:max_return]]


def recommend_posts(
    graph: nx.Graph,
    target_user_id: str,
    db_path: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    max_return: int = 3
) -> List[Dict]:
    """
    Recommend posts for a target user using the provided hot score formula.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row  # Return results as dictionaries

            # Step 1: Retrieve posts from all users except target user
            query = '''
                SELECT p.id, p.user_id, p.content, p.score, p.belief, p.type, p.created_at
                FROM posts p
                WHERE p.user_id != ? AND p.created_at BETWEEN ? AND ?
            '''
            params = (target_user_id, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            cursor = conn.execute(query, params)
            all_posts = cursor.fetchall()

            # Step 2: For each post, calculate the number of upvotes and downvotes
            ranked_posts = []
            t0 = 1134028003  # Constant from the formula

            for post in all_posts:
                # Step 2.1: Count upvotes and downvotes for this post
                post_id = post["id"]
                reactions_query = '''
                    SELECT score, COUNT(*) AS count
                    FROM reactions
                    WHERE post_id = ?
                    GROUP BY score
                '''
                reactions_cursor = conn.execute(reactions_query, (post_id,))
                reactions = reactions_cursor.fetchall()

                # Default counts for upvotes and downvotes
                upvotes = 0
                downvotes = 0

                # Fill upvote and downvote counts based on the reactions
                for reaction in reactions:
                    if reaction["score"] == 1:
                        upvotes = reaction["count"]
                    elif reaction["score"] == -1:
                        downvotes = reaction["count"]

                # Step 2.2: Calculate the hot score
                u = upvotes  # Number of upvotes
                d = downvotes  # Number of downvotes
                t = pd.to_datetime(post["created_at"]).timestamp()  # Submission time in seconds

                # Calculate the hot score
                h = math.log10(max(abs(u - d), 1)) + math.copysign(1, u - d) * (t - t0) / 45000
                ranked_posts.append((h, dict(post)))  # Convert sqlite3.Row to dict

            # Step 3: Sort posts by hot score in descending order
            ranked_posts.sort(key=lambda x: x[0], reverse=True)

            # Step 4: Return the top posts
            return [post for _, post in ranked_posts[:max_return]]

    except sqlite3.Error as e:
        print(f"Error recommending posts: {e}")
        return []


def recommend_post_graph(
    graph: nx.Graph,
    target_user_id: str,
    db_path: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    max_return: int = 3
) -> List[Dict]:
    """
    基于社交网络图的帖子推荐算法
    
    该函数实现了一个基于用户社交关系的智能帖子推荐系统。
    只推荐来自用户直接社交连接的帖子，并使用改进的热度算法进行排序。
    
    推荐算法特性：
    1. 社交过滤：只推荐来自图中直接邻居用户的帖子
    2. 时间衰减：考虑帖子发布时间对热度的影响
    3. 对数缩放：使用对数函数处理互动数量，避免极值影响
    4. 热度排序：基于改进的热度公式进行帖子排序
    5. 质量筛选：自动过滤低质量和过时内容
    
    热度计算公式：
    - 正向内容：log10(净点赞数 + 1) / (时间衰减因子 ^ 1.8)
    - 负向内容：-时间衰减值（快速下沉）
    
    Args:
        graph (nx.Graph): 用户社交关系网络图
        target_user_id (str): 目标用户ID（推荐对象）
        db_path (str): 论坛数据库文件路径
        start_date (pd.Timestamp): 推荐内容的开始时间
        end_date (pd.Timestamp): 推荐内容的结束时间
        max_return (int): 最大返回帖子数量，默认3个
        
    Returns:
        List[Dict]: 推荐的帖子列表，按热度降序排列，每个帖子包含：
            - id: 帖子ID
            - user_id: 发帖用户ID
            - content: 帖子内容
            - score: 当前评分
            - belief: 用户信念值
            - type: 帖子类型
            - created_at: 创建时间
            
    Note:
        - 只推荐来自社交网络邻居的帖子
        - 使用改进的热度算法确保内容质量
        - 支持时间范围的灵活配置
        - 自动处理无邻居用户的情况
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row  # Return results as dictionaries

            # Step 1: Get all users that are neighbors of the target user in the graph
            neighbors = list(graph.neighbors(target_user_id))

            if not neighbors:
                return []

            # Step 2: Retrieve posts from these neighboring users
            query = '''
                SELECT p.id, p.user_id, p.content, p.score, p.belief, p.type, p.created_at
                FROM posts p
                WHERE p.user_id IN ({}) AND p.created_at BETWEEN ? AND ?
            '''.format(','.join(['?'] * len(neighbors)))  # Create a parameterized IN clause
            params = neighbors + [start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')]
            cursor = conn.execute(query, params)
            all_posts = cursor.fetchall()

            # Step 3: For each post, calculate the number of upvotes and downvotes
            ranked_posts = []
            current_time = pd.Timestamp.now().timestamp()  # Get current timestamp

            for post in all_posts:
                # Step 3.1: Count upvotes and downvotes for this post
                post_id = post["id"]
                reactions_query = '''
                    SELECT type, COUNT(*) AS count
                    FROM reactions
                    WHERE post_id = ?
                    GROUP BY type
                '''
                reactions_cursor = conn.execute(reactions_query, (post_id,))
                reactions = reactions_cursor.fetchall()

                # Default counts for upvotes and downvotes
                upvotes = 0
                downvotes = 0

                # Fill upvote and downvote counts based on the reactions
                for reaction in reactions:
                    if reaction["type"] == 'like':
                        upvotes = reaction["count"]
                    elif reaction["type"] == 'unlike':
                        downvotes = reaction["count"]

                # Step 3.2: Calculate the hot score
                net_votes = upvotes - downvotes
                post_time = pd.to_datetime(post["created_at"]).timestamp()
                days_elapsed = max((current_time - post_time) / 86400, 0.1)  # Prevent division by zero

                if net_votes > 0:
                    score = math.log10(net_votes + 1)  # Logarithmic scaling of net votes
                    time_decay = (days_elapsed + 1) ** 1.8  # Time decay factor
                    hot_score = score / time_decay  # Final hot score
                else:
                    hot_score = -days_elapsed  # Negative content sinks quickly

                ranked_posts.append((hot_score, dict(post)))  # Convert sqlite3.Row to dict

            # Step 4: Sort posts by hot score in descending order
            ranked_posts.sort(key=lambda x: x[0], reverse=True)

            # Step 5: Return the top posts
            return [post for _, post in ranked_posts[:max_return]]

    except sqlite3.Error as e:
        print(f"Error recommending posts: {e}")
        return []


# def get_all_recommend_posts(
#     graph: nx.Graph,
#     db_path: str,
#     start_date: pd.Timestamp,
#     end_date: pd.Timestamp,
#     max_return: int = 3
# ) -> Dict[str, List[Dict]]:
#     # Step 1: Compute PageRank scores for all users in the graph
#     pagerank_scores = nx.pagerank(graph, weight="weight")

#     # Step 2: Retrieve posts from all users
#     all_posts = []
#     for user_id in graph.nodes():
#         posts = get_user_posts_db(
#             user_id=user_id,
#             db_path=db_path,
#             start_date=start_date,
#             end_date=end_date
#         )
#         all_posts.extend(posts)

#     # Step 3: Rank posts based on PageRank, like_score, and timestamp
#     ranked_posts = []
#     for post in all_posts:
#         post_score = (
#             pagerank_scores.get(post["user_id"], 0) * 0.5 +  # Weight for user influence
#             post["like_score"] * 0.3 +  # Weight for post engagement
#             (pd.to_datetime(post["created_at"]).timestamp() / 1e9) * 0.2  # Weight for recency (normalized timestamp)
#         )
#         ranked_posts.append((post_score, post))

#     # Sort posts by score in descending order
#     ranked_posts.sort(key=lambda x: x[0], reverse=True)

#     # Step 4: Group posts by user ID
#     user_recommendations = {}
#     for _, post in ranked_posts:
#         user_id = post["user_id"]
#         if user_id not in user_recommendations:
#             user_recommendations[user_id] = []
#         if len(user_recommendations[user_id]) < max_return:
#             user_recommendations[user_id].append(post)

#     return user_recommendations


def get_user_reactions_db(
    user_id: str,
    reaction_type: Optional[str] = None,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    db_path: str = FORUM_DB_PATH
) -> List[Dict]:
    """
    Get all reactions by a user within date range

    Args:
        user_id (str): User ID to get reactions for
        reaction_type (Optional[str]): Filter by 'like' or 'unlike'
        start_date (Optional[pd.Timestamp]): Start of date range
        end_date (Optional[pd.Timestamp]): End of date range
        db_path (str): Database path

    Returns:
        list[dict]: List of reaction dictionaries with keys:
            - post_id: ID of post reacted to
            - type: reaction type (like/unlike)
            - created_at: reaction timestamp
            - post_content: content of post reacted to
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = """
                SELECT r.*, p.content as post_content
                FROM reactions r
                JOIN posts p ON r.post_id = p.id
                WHERE r.user_id = ?
            """
            params = [user_id]

            if reaction_type:
                query += " AND r.score = ?"
                params.append(1 if reaction_type == "like" else -1)
            if start_date:
                query += " AND r.created_at >= ?"
                params.append(start_date.strftime('%Y-%m-%d'))
            if end_date:
                query += " AND r.created_at <= ?"
                params.append(end_date.strftime('%Y-%m-%d'))

            query += " ORDER BY r.created_at DESC"

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    except sqlite3.Error as e:
        print(f"Error getting user reactions: {e}")
        return []


def get_post_by_id_db(
    post_id: int,
    db_path: str = FORUM_DB_PATH
) -> Optional[Dict]:
    """
    Get detailed information about a specific post

    Args:
        post_id (int): ID of the post to retrieve
        db_path (str): Database path

    Returns:
        Optional[dict]: Post information including:
            - id: post ID
            - user_id: poster's user ID
            - content: post content
            - created_at: post timestamp
            - score: current post score
            - likes_count: number of positive reactions
            - unlikes_count: number of negative reactions
            - reference_id: ID of referenced post (if repost)
            Or None if post not found
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = """
                SELECT 
                    p.*,
                    COUNT(CASE WHEN r.score = 1 THEN 1 END) as likes_count,
                    COUNT(CASE WHEN r.score = -1 THEN 1 END) as unlikes_count,
                    pr.reference_id
                FROM posts p
                LEFT JOIN reactions r ON p.id = r.post_id
                LEFT JOIN post_references pr ON p.id = pr.id
                WHERE p.id = ?
                GROUP BY p.id
            """

            cursor = conn.execute(query, (post_id,))
            row = cursor.fetchone()

            if row is None:
                return None

            return dict(row)

    except sqlite3.Error as e:
        print(f"Error getting post {post_id}: {e}")
        return None


def get_post_count_by_date_range_db(
    start_date: str,
    end_date: str,
    db_path: str = FORUM_DB_PATH
) -> Tuple[int, int]:
    """
    Get total post count and unique user count within a date range.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        db_path (str): Path to the database. Defaults to FORUM_DB_PATH.

    Returns:
        Tuple[int, int]: 
            First int: Total post count in date range
            Second int: Total unique user count in date range
    """
    try:
        with sqlite3.connect(db_path) as conn:
            query = """
                SELECT 
                    COUNT(*) as total_posts,
                    COUNT(DISTINCT user_id) as unique_users
                FROM posts
                WHERE created_at >= ? AND created_at <= ?
            """

            cursor = conn.execute(query, (start_date, end_date))
            result = cursor.fetchone()

            total_posts = result[0] if result else 0
            unique_users = result[1] if result else 0

            return total_posts, unique_users

    except sqlite3.Error as e:
        print(f"Error getting stats: {e}")
        return 0, 0
    

def find_root_post(post_id, db_path):
    """
    根据 post_id 和数据库路径，找到该帖子的根帖子。
    首先从 post_references 表中查找引用路径，然后从 posts 表中查找根帖子。

    Args:
        post_id (int): 当前帖子的 ID。
        db_path (str): 数据库路径。

    Returns:
        dict: 根帖子的详细信息，包括 post_id, user_id, created_at, score, content。
              如果找不到根帖子，则返回 None。
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # 初始化路径追踪
        current_post_id = post_id
        path = []

        while True:
            # 查找当前帖子的父帖子 ID
            cursor.execute('''
                SELECT reference_id
                FROM post_references
                WHERE repost_id = ?
            ''', (current_post_id,))
            row = cursor.fetchone()

            if not row:
                break  # 如果没有找到父帖子，则停止查找

            path.append(current_post_id)  # 记录路径
            current_post_id = row[0]  # 更新为父帖子 ID

        # 最后一个有效的帖子 ID 应该是根帖子的候选
        root_candidate_id = current_post_id

        # 在 posts 表中查询根帖子的详细信息
        cursor.execute('''
            SELECT
                p.id,
                p.user_id,
                p.created_at,
                p.type,
                p.score,
                p.content
            FROM posts p
            WHERE p.id = ?
        ''', (root_candidate_id,))
        root_row = cursor.fetchone()

        if not root_row or root_row[3] == 'repost':
            return None  # 如果根帖子不存在或类型为 repost，则返回 None

        return {
            "post_id": root_row[0],
            "user_id": root_row[1],
            "created_at": root_row[2],
            "score": root_row[4],
            "content": root_row[5],
            "path": path  # 返回完整路径
        }

# init_db_forum('/home/export/base/ycsc_wangbenyou/yangyz/online1/toby/Graph-Agent-Network/TwinMarket/data/ForumDB/sys_1000.db')