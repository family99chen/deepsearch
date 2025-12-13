"""
MongoDB 缓存模块

提供简单的接口用于缓存 API 调用结果，节省 API 成本。

使用示例:
    from localdb.insert_mongo import MongoCache
    
    # 方式1: 使用默认配置
    cache = MongoCache()
    
    # 方式2: 指定数据库和集合
    cache = MongoCache(db_name="deepsearch", collection_name="orcid_cache")
    
    # 设置缓存（永久）
    cache.set("orcid:0000-0001-2345-6789", {"name": "John Doe", "papers": [...]})
    
    # 设置缓存（带过期时间，单位：秒）
    cache.set("orcid:0000-0001-2345-6789", data, ttl=86400)  # 24小时过期
    
    # 获取缓存
    data = cache.get("orcid:0000-0001-2345-6789")
    if data:
        print("命中缓存")
    
    # 删除缓存
    cache.delete("orcid:0000-0001-2345-6789")
    
    # 检查是否存在
    if cache.exists("orcid:0000-0001-2345-6789"):
        ...
    
    # 使用装饰器自动缓存函数结果
    @cache.cached(key_prefix="orcid_person", ttl=3600)
    def fetch_orcid_person(orcid_id):
        # 这个函数的结果会被自动缓存
        return api_call(orcid_id)
"""

import sys
import yaml
import hashlib
import functools
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Any, Dict, Callable, Union

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    HAS_PYMONGO = True
except ImportError:
    HAS_PYMONGO = False
    print("[WARNING] pymongo 未安装，缓存功能不可用")
    print("[TIP] 安装: pip install pymongo")


def load_config() -> dict:
    """加载配置文件"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class MongoCache:
    """
    MongoDB 缓存管理器
    
    支持:
    - 基本的 get/set/delete 操作
    - 自动过期 (TTL)
    - 函数结果缓存装饰器
    - 批量操作
    """
    
    # 默认配置
    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 27017
    DEFAULT_DB = "deepsearch_cache"
    DEFAULT_COLLECTION = "api_cache"
    DEFAULT_TTL = None  # 永不过期
    
    def __init__(
        self,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        connection_timeout: int = 5000,
    ):
        """
        初始化 MongoDB 缓存
        
        Args:
            db_name: 数据库名称，默认从配置读取或使用 'deepsearch_cache'
            collection_name: 集合名称，默认从配置读取或使用 'api_cache'
            host: MongoDB 主机地址
            port: MongoDB 端口
            username: 用户名（可选）
            password: 密码（可选）
            connection_timeout: 连接超时（毫秒）
        """
        if not HAS_PYMONGO:
            raise ImportError("pymongo 未安装，请运行: pip install pymongo")
        
        # 加载配置
        config = load_config()
        mongo_config = config.get("mongodb", {})
        
        # 使用配置文件或参数或默认值
        self.host = host or mongo_config.get("host", self.DEFAULT_HOST)
        self.port = port or mongo_config.get("port", self.DEFAULT_PORT)
        self.db_name = db_name or mongo_config.get("db_name", self.DEFAULT_DB)
        self.collection_name = collection_name or mongo_config.get("collection", self.DEFAULT_COLLECTION)
        self.username = username or mongo_config.get("username")
        self.password = password or mongo_config.get("password")
        self.auth_source = mongo_config.get("auth_source", "admin")  # 默认 admin
        
        # 构建连接 URI
        if self.username and self.password:
            uri = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/?authSource={self.auth_source}"
        else:
            uri = f"mongodb://{self.host}:{self.port}/"
        
        # 连接 MongoDB
        try:
            self.client = MongoClient(
                uri,
                serverSelectionTimeoutMS=connection_timeout,
                connectTimeoutMS=connection_timeout,
            )
            # 测试连接
            self.client.admin.command('ping')
            self._connected = True
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"[WARNING] MongoDB 连接失败: {e}")
            print(f"[INFO] 缓存功能将被禁用，所有操作将直接返回 None")
            self._connected = False
            self.client = None
        
        if self._connected:
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            
            # 创建索引
            self._ensure_indexes()
    
    def _ensure_indexes(self):
        """确保必要的索引存在"""
        if not self._connected:
            return
        
        try:
            # 键的唯一索引
            self.collection.create_index("key", unique=True)
            # 过期时间索引（MongoDB 会自动删除过期文档）
            self.collection.create_index("expire_at", expireAfterSeconds=0)
        except Exception as e:
            print(f"[WARNING] 创建索引失败: {e}")
    
    def is_connected(self) -> bool:
        """检查是否已连接到 MongoDB"""
        return self._connected
    
    def use_collection(self, collection_name: str) -> "MongoCache":
        """
        切换到指定集合（返回新实例）
        
        Args:
            collection_name: 集合名称
            
        Returns:
            新的 MongoCache 实例
        """
        return MongoCache(
            db_name=self.db_name,
            collection_name=collection_name,
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
        )
    
    def use_db(self, db_name: str, collection_name: Optional[str] = None) -> "MongoCache":
        """
        切换到指定数据库（返回新实例）
        
        Args:
            db_name: 数据库名称
            collection_name: 集合名称（可选）
            
        Returns:
            新的 MongoCache 实例
        """
        return MongoCache(
            db_name=db_name,
            collection_name=collection_name or self.collection_name,
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
        )
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值（会被序列化存储）
            ttl: 过期时间（秒），None 表示永不过期
            metadata: 额外的元数据
            
        Returns:
            是否成功
        """
        if not self._connected:
            return False
        
        try:
            doc = {
                "key": key,
                "value": value,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
            
            # 设置过期时间
            if ttl is not None and ttl > 0:
                doc["expire_at"] = datetime.utcnow() + timedelta(seconds=ttl)
                doc["ttl"] = ttl
            else:
                # 移除过期时间（永不过期）
                doc["expire_at"] = None
                doc["ttl"] = None
            
            # 添加元数据
            if metadata:
                doc["metadata"] = metadata
            
            # 使用 upsert 实现插入或更新
            self.collection.update_one(
                {"key": key},
                {"$set": doc},
                upsert=True
            )
            return True
            
        except Exception as e:
            print(f"[ERROR] 缓存写入失败: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存
        
        Args:
            key: 缓存键
            default: 缓存不存在时的默认值
            
        Returns:
            缓存值或默认值
        """
        if not self._connected:
            return default
        
        try:
            doc = self.collection.find_one({"key": key})
            
            if doc is None:
                return default
            
            # 检查是否过期（手动检查，以防 TTL 索引延迟）
            expire_at = doc.get("expire_at")
            if expire_at and datetime.utcnow() > expire_at:
                # 已过期，删除并返回默认值
                self.delete(key)
                return default
            
            return doc.get("value", default)
            
        except Exception as e:
            print(f"[ERROR] 缓存读取失败: {e}")
            return default
    
    def set_field(
        self,
        key: str,
        field_name: str,
        field_value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        设置/更新文档中的特定字段（用于同一个 key 存储多个字段）
        
        Args:
            key: 缓存键
            field_name: 字段名（会存储在 value.field_name 下）
            field_value: 字段值
            ttl: 过期时间（秒），None 表示永不过期
            
        Returns:
            是否成功
            
        Usage:
            # 同一个 ORCID 存储多个信息
            cache.set_field("orcid:0000-0001", "person_info", {"name": "John"})
            cache.set_field("orcid:0000-0001", "papers", [...])
            cache.set_field("orcid:0000-0001", "organizations", [...])
            
            # 结果文档：
            # {
            #   "key": "orcid:0000-0001",
            #   "value": {
            #     "person_info": {"name": "John"},
            #     "papers": [...],
            #     "organizations": [...]
            #   }
            # }
        """
        if not self._connected:
            return False
        
        try:
            update_doc = {
                f"value.{field_name}": field_value,
                "updated_at": datetime.utcnow(),
            }
            
            # 设置过期时间
            if ttl is not None and ttl > 0:
                update_doc["expire_at"] = datetime.utcnow() + timedelta(seconds=ttl)
                update_doc["ttl"] = ttl
            
            # 使用 upsert + $set 实现字段更新
            self.collection.update_one(
                {"key": key},
                {
                    "$set": update_doc,
                    "$setOnInsert": {
                        "key": key,
                        "created_at": datetime.utcnow(),
                    }
                },
                upsert=True
            )
            return True
            
        except Exception as e:
            print(f"[ERROR] 字段更新失败: {e}")
            return False
    
    def get_field(self, key: str, field_name: str, default: Any = None) -> Any:
        """
        获取文档中的特定字段
        
        Args:
            key: 缓存键
            field_name: 字段名
            default: 字段不存在时的默认值
            
        Returns:
            字段值或默认值
            
        Usage:
            person_info = cache.get_field("orcid:0000-0001", "person_info")
            papers = cache.get_field("orcid:0000-0001", "papers")
        """
        if not self._connected:
            return default
        
        try:
            doc = self.collection.find_one({"key": key})
            
            if doc is None:
                return default
            
            # 检查是否过期
            expire_at = doc.get("expire_at")
            if expire_at and datetime.utcnow() > expire_at:
                self.delete(key)
                return default
            
            value = doc.get("value", {})
            if isinstance(value, dict):
                return value.get(field_name, default)
            
            return default
            
        except Exception as e:
            print(f"[ERROR] 字段读取失败: {e}")
            return default
    
    def has_field(self, key: str, field_name: str) -> bool:
        """
        检查文档是否存在特定字段
        
        Args:
            key: 缓存键
            field_name: 字段名
            
        Returns:
            是否存在该字段
        """
        if not self._connected:
            return False
        
        try:
            doc = self.collection.find_one(
                {"key": key, f"value.{field_name}": {"$exists": True}},
                {"_id": 1, "expire_at": 1}
            )
            
            if doc is None:
                return False
            
            # 检查是否过期
            expire_at = doc.get("expire_at")
            if expire_at and datetime.utcnow() > expire_at:
                self.delete(key)
                return False
            
            return True
            
        except Exception as e:
            print(f"[ERROR] 字段检查失败: {e}")
            return False
    
    def get_with_metadata(self, key: str) -> Optional[Dict]:
        """
        获取缓存及其元数据
        
        Args:
            key: 缓存键
            
        Returns:
            包含 value, metadata, created_at, ttl 等信息的字典，不存在则返回 None
        """
        if not self._connected:
            return None
        
        try:
            doc = self.collection.find_one({"key": key})
            
            if doc is None:
                return None
            
            # 检查是否过期
            expire_at = doc.get("expire_at")
            if expire_at and datetime.utcnow() > expire_at:
                self.delete(key)
                return None
            
            return {
                "value": doc.get("value"),
                "metadata": doc.get("metadata"),
                "created_at": doc.get("created_at"),
                "updated_at": doc.get("updated_at"),
                "ttl": doc.get("ttl"),
                "expire_at": doc.get("expire_at"),
            }
            
        except Exception as e:
            print(f"[ERROR] 缓存读取失败: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            是否成功删除
        """
        if not self._connected:
            return False
        
        try:
            result = self.collection.delete_one({"key": key})
            return result.deleted_count > 0
        except Exception as e:
            print(f"[ERROR] 缓存删除失败: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        检查缓存是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            是否存在
        """
        if not self._connected:
            return False
        
        try:
            doc = self.collection.find_one({"key": key}, {"_id": 1, "expire_at": 1})
            
            if doc is None:
                return False
            
            # 检查是否过期
            expire_at = doc.get("expire_at")
            if expire_at and datetime.utcnow() > expire_at:
                self.delete(key)
                return False
            
            return True
            
        except Exception as e:
            print(f"[ERROR] 缓存检查失败: {e}")
            return False
    
    def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None) -> int:
        """
        批量设置缓存
        
        Args:
            items: 键值对字典
            ttl: 过期时间（秒）
            
        Returns:
            成功设置的数量
        """
        if not self._connected:
            return 0
        
        success_count = 0
        for key, value in items.items():
            if self.set(key, value, ttl=ttl):
                success_count += 1
        return success_count
    
    def get_many(self, keys: list) -> Dict[str, Any]:
        """
        批量获取缓存
        
        Args:
            keys: 键列表
            
        Returns:
            键值对字典（只包含存在的键）
        """
        if not self._connected:
            return {}
        
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    def delete_many(self, keys: list) -> int:
        """
        批量删除缓存
        
        Args:
            keys: 键列表
            
        Returns:
            成功删除的数量
        """
        if not self._connected:
            return 0
        
        try:
            result = self.collection.delete_many({"key": {"$in": keys}})
            return result.deleted_count
        except Exception as e:
            print(f"[ERROR] 批量删除失败: {e}")
            return 0
    
    def clear(self, pattern: Optional[str] = None) -> int:
        """
        清空缓存
        
        Args:
            pattern: 可选的键前缀模式（如 "orcid:*"）
            
        Returns:
            删除的数量
        """
        if not self._connected:
            return 0
        
        try:
            if pattern:
                # 使用正则匹配
                import re
                regex_pattern = pattern.replace("*", ".*")
                result = self.collection.delete_many({"key": {"$regex": f"^{regex_pattern}"}})
            else:
                result = self.collection.delete_many({})
            return result.deleted_count
        except Exception as e:
            print(f"[ERROR] 清空缓存失败: {e}")
            return 0
    
    def count(self, pattern: Optional[str] = None) -> int:
        """
        统计缓存数量
        
        Args:
            pattern: 可选的键前缀模式
            
        Returns:
            缓存数量
        """
        if not self._connected:
            return 0
        
        try:
            if pattern:
                regex_pattern = pattern.replace("*", ".*")
                return self.collection.count_documents({"key": {"$regex": f"^{regex_pattern}"}})
            return self.collection.count_documents({})
        except Exception as e:
            print(f"[ERROR] 统计失败: {e}")
            return 0
    
    def keys(self, pattern: Optional[str] = None, limit: int = 100) -> list:
        """
        获取所有缓存键
        
        Args:
            pattern: 可选的键前缀模式
            limit: 最大返回数量
            
        Returns:
            键列表
        """
        if not self._connected:
            return []
        
        try:
            query = {}
            if pattern:
                regex_pattern = pattern.replace("*", ".*")
                query["key"] = {"$regex": f"^{regex_pattern}"}
            
            cursor = self.collection.find(query, {"key": 1}).limit(limit)
            return [doc["key"] for doc in cursor]
        except Exception as e:
            print(f"[ERROR] 获取键列表失败: {e}")
            return []
    
    def cached(
        self,
        key_prefix: str = "",
        ttl: Optional[int] = None,
        key_builder: Optional[Callable] = None,
    ):
        """
        函数结果缓存装饰器
        
        Args:
            key_prefix: 缓存键前缀
            ttl: 过期时间（秒）
            key_builder: 自定义键生成函数，接收函数参数，返回键字符串
            
        Returns:
            装饰器
            
        Usage:
            @cache.cached(key_prefix="orcid", ttl=3600)
            def fetch_orcid_data(orcid_id):
                return api_call(orcid_id)
            
            # 自定义键生成
            @cache.cached(key_builder=lambda orcid, **kw: f"orcid:{orcid}")
            def fetch_data(orcid_id, force=False):
                ...
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存键
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    # 默认键生成：前缀 + 函数名 + 参数哈希
                    key_parts = [key_prefix, func.__name__]
                    if args:
                        key_parts.append(str(args))
                    if kwargs:
                        key_parts.append(str(sorted(kwargs.items())))
                    
                    raw_key = ":".join(filter(None, key_parts))
                    # 如果键太长，使用哈希
                    if len(raw_key) > 200:
                        cache_key = f"{key_prefix}:{func.__name__}:{hashlib.md5(raw_key.encode()).hexdigest()}"
                    else:
                        cache_key = raw_key
                
                # 检查 force 参数，用于强制刷新缓存
                force_refresh = kwargs.pop('force_refresh', False)
                
                if not force_refresh:
                    # 尝试从缓存获取
                    cached_value = self.get(cache_key)
                    if cached_value is not None:
                        return cached_value
                
                # 调用原函数
                result = func(*args, **kwargs)
                
                # 存入缓存
                if result is not None:
                    self.set(cache_key, result, ttl=ttl)
                
                return result
            
            # 添加手动操作缓存的方法
            wrapper.cache_key = lambda *args, **kwargs: (
                key_builder(*args, **kwargs) if key_builder 
                else f"{key_prefix}:{func.__name__}:{args}:{sorted(kwargs.items())}"
            )
            wrapper.invalidate = lambda *args, **kwargs: self.delete(wrapper.cache_key(*args, **kwargs))
            
            return wrapper
        return decorator
    
    def close(self):
        """关闭 MongoDB 连接"""
        if self.client:
            self.client.close()
            self._connected = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def stats(self) -> Dict:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        if not self._connected:
            return {"connected": False}
        
        try:
            return {
                "connected": True,
                "db_name": self.db_name,
                "collection_name": self.collection_name,
                "total_count": self.collection.count_documents({}),
                "db_stats": self.db.command("dbStats"),
            }
        except Exception as e:
            return {"connected": True, "error": str(e)}


# ============ 便捷函数 ============

# 全局缓存实例（懒加载）
_global_cache: Optional[MongoCache] = None


def get_cache(db_name: Optional[str] = None, collection_name: Optional[str] = None) -> MongoCache:
    """
    获取缓存实例（单例模式）
    
    Args:
        db_name: 数据库名称
        collection_name: 集合名称
        
    Returns:
        MongoCache 实例
    """
    global _global_cache
    
    if _global_cache is None:
        _global_cache = MongoCache(db_name=db_name, collection_name=collection_name)
    elif db_name or collection_name:
        # 如果指定了不同的 db/collection，返回新实例
        if (db_name and db_name != _global_cache.db_name) or \
           (collection_name and collection_name != _global_cache.collection_name):
            return MongoCache(db_name=db_name, collection_name=collection_name)
    
    return _global_cache


def cache_set(key: str, value: Any, ttl: Optional[int] = None, 
              collection: Optional[str] = None) -> bool:
    """
    便捷函数：设置缓存
    
    Args:
        key: 缓存键
        value: 缓存值
        ttl: 过期时间（秒）
        collection: 集合名称（可选）
    """
    cache = get_cache(collection_name=collection)
    return cache.set(key, value, ttl=ttl)


def cache_get(key: str, default: Any = None, collection: Optional[str] = None) -> Any:
    """
    便捷函数：获取缓存
    
    Args:
        key: 缓存键
        default: 默认值
        collection: 集合名称（可选）
    """
    cache = get_cache(collection_name=collection)
    return cache.get(key, default)


def cache_delete(key: str, collection: Optional[str] = None) -> bool:
    """
    便捷函数：删除缓存
    
    Args:
        key: 缓存键
        collection: 集合名称（可选）
    """
    cache = get_cache(collection_name=collection)
    return cache.delete(key)


def cache_exists(key: str, collection: Optional[str] = None) -> bool:
    """
    便捷函数：检查缓存是否存在
    
    Args:
        key: 缓存键
        collection: 集合名称（可选）
    """
    cache = get_cache(collection_name=collection)
    return cache.exists(key)


# ============ 测试代码 ============

if __name__ == "__main__":
    print("=" * 60)
    print("MongoDB 缓存模块测试")
    print("=" * 60)
    
    # 创建缓存实例
    cache = MongoCache()
    
    if not cache.is_connected():
        print("[ERROR] 无法连接到 MongoDB，请确保 MongoDB 正在运行")
        print("[TIP] 启动 MongoDB: sudo systemctl start mongod")
        sys.exit(1)
    
    print(f"[OK] 已连接到 MongoDB")
    print(f"[INFO] 数据库: {cache.db_name}")
    print(f"[INFO] 集合: {cache.collection_name}")
    print()
    
    # 测试基本操作
    print("--- 测试基本操作 ---")
    
    # 设置缓存
    test_key = "test:user:12345"
    test_value = {"name": "John Doe", "email": "john@example.com", "papers": [1, 2, 3]}
    
    print(f"[SET] key={test_key}")
    cache.set(test_key, test_value)
    
    # 获取缓存
    result = cache.get(test_key)
    print(f"[GET] value={result}")
    
    # 检查存在
    print(f"[EXISTS] {cache.exists(test_key)}")
    
    # 删除缓存
    cache.delete(test_key)
    print(f"[DELETE] key={test_key}")
    print(f"[EXISTS after delete] {cache.exists(test_key)}")
    print()
    
    # 测试 TTL
    print("--- 测试 TTL (3秒过期) ---")
    cache.set("test:ttl", "这条数据3秒后过期", ttl=3)
    print(f"[SET with TTL] 已设置")
    print(f"[GET immediately] {cache.get('test:ttl')}")
    
    import time
    print("[WAIT] 等待4秒...")
    time.sleep(4)
    print(f"[GET after 4s] {cache.get('test:ttl')}")
    print()
    
    # 测试装饰器
    print("--- 测试装饰器缓存 ---")
    
    call_count = 0
    
    @cache.cached(key_prefix="test_func", ttl=60)
    def expensive_function(x, y):
        global call_count
        call_count += 1
        print(f"  [函数执行] 第 {call_count} 次调用")
        return x + y
    
    print("[CALL 1] expensive_function(1, 2)")
    result1 = expensive_function(1, 2)
    print(f"  结果: {result1}")
    
    print("[CALL 2] expensive_function(1, 2) - 应该命中缓存")
    result2 = expensive_function(1, 2)
    print(f"  结果: {result2}")
    
    print(f"[INFO] 函数实际执行次数: {call_count}")
    print()
    
    # 统计信息
    print("--- 缓存统计 ---")
    stats = cache.stats()
    print(f"数据库: {stats.get('db_name')}")
    print(f"集合: {stats.get('collection_name')}")
    print(f"缓存条目数: {stats.get('total_count')}")
    print()
    
    # 清理测试数据
    print("--- 清理测试数据 ---")
    deleted = cache.clear("test:*")
    print(f"已清理 {deleted} 条测试数据")
    deleted = cache.clear("test_func:*")
    print(f"已清理 {deleted} 条装饰器测试数据")
    
    print()
    print("=" * 60)
    print("测试完成！")
    print("=" * 60)

